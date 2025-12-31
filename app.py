#!/usr/bin/env python3
"""
Enhanced Telegram Document Tutor
- Accepts PDF / DOCX with improved error handling
- Extracts text + images with better quality
- RAG with SentenceTransformers + FAISS
- Uses Gemini for teaching and Q&A
- Added conversation history, summaries, and quiz features
"""

import os
import uuid
import time 
import asyncio
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

# 3rd-party libs
try:
    import fitz  # PyMuPDF
    import docx
    import google.generativeai as genai
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    from dotenv import load_dotenv
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import (
        ApplicationBuilder,
        CommandHandler,
        MessageHandler,
        CallbackQueryHandler,
        ContextTypes,
        filters,
    )
except ImportError as e:
    raise SystemExit(f"Missing dependency: {e}\nRun: pip install pymupdf python-docx google-generativeai sentence-transformers faiss-cpu numpy python-dotenv python-telegram-bot")

# ----------------- CONFIG / ENV -----------------
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "20"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN in env or .env")
if not GEMINI_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY in env or .env")

genai.configure(api_key=GEMINI_API_KEY)

# Global embedding model (lazy loaded)
EMBED_MODEL_OBJ = None

# Directories
DOWNLOAD_DIR = "downloads"
IMG_DIR = "extracted_images"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("doc_tutor")

# ----------------- DATA STRUCTURES -----------------
@dataclass
class Section:
    section_id: str
    heading: str
    content: str
    location: str
    images: List[str] = field(default_factory=list)
    summary: Optional[str] = None

@dataclass
class Image:
    image_id: str
    path: str
    page: int
    context_text: str

@dataclass
class UserState:
    sections: List[Section] = field(default_factory=list)
    images: List[Image] = field(default_factory=list)
    rag: Optional['RAGIndex'] = None
    document_name: str = ""
    upload_time: datetime = field(default_factory=datetime.now)
    conversation_history: List[Dict] = field(default_factory=list)
    current_section: int = 0
    pending_images: List[str] = field(default_factory=list)

# In-memory state
USER_STATE: Dict[int, UserState] = {}

# ----------------- UTILITY HELPERS -----------------
async def run_blocking(fn, *args, **kwargs):
    """Run blocking operations in thread pool"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

def chunk_text_for_telegram(text: str, max_chars: int = 4000) -> List[str]:
    """
    Break long text into chunks respecting Telegram's 4096 char limit.
    Tries to split at paragraph boundaries.
    """
    if len(text) <= max_chars:
        return [text]
    
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        # Fallback: split by sentences
        paragraphs = [s.strip() for s in text.split(". ") if s.strip()]
    
    chunks = []
    current = []
    current_len = 0
    
    for para in paragraphs:
        para_len = len(para) + 2  # +2 for "\n\n"
        
        if current_len + para_len > max_chars and current:
            chunks.append("\n\n".join(current))
            current = [para]
            current_len = para_len
        elif para_len > max_chars:
            # Single paragraph too long - force split
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            
            for i in range(0, len(para), max_chars):
                chunks.append(para[i:i + max_chars])
        else:
            current.append(para)
            current_len += para_len
    
    if current:
        chunks.append("\n\n".join(current))
    
    return chunks

async def safe_send_text(bot, chat_id: int, text: str):
    """Send text with automatic chunking if needed"""
    chunks = chunk_text_for_telegram(text)
    for i, chunk in enumerate(chunks):
        try:
            await bot.send_message(chat_id=chat_id, text=chunk, parse_mode='Markdown')
        except Exception:
            # Fallback without markdown if parsing fails
            try:
                await bot.send_message(chat_id=chat_id, text=chunk)
            except Exception as e:
                logger.error(f"Failed to send message: {e}")
        
        # Small delay between chunks to avoid rate limiting
        if i < len(chunks) - 1:
            await asyncio.sleep(0.5)

# ----------------- EXTRACTORS -----------------
def _save_pixmap(pix, out_path: str) -> bool:
    """Save a fitz.Pixmap safely, returns success status"""
    try:
        if pix.n < 5:  # RGB or Gray
            pix.save(out_path)
        else:  # CMYK or with alpha
            new_pix = fitz.Pixmap(fitz.csRGB, pix)
            new_pix.save(out_path)
            new_pix = None
        return True
    except Exception as e:
        logger.warning(f"Failed to save pixmap: {e}")
        return False
    finally:
        pix = None

def extract_pdf(path: str) -> Tuple[List[Section], List[Image]]:
    """
    Extract page-wise sections and images from PDF with improved handling.
    """
    try:
        doc = fitz.open(path)
    except Exception as e:
        raise ValueError(f"Cannot open PDF: {e}")
    
    sections = []
    images = []

    for pno, page in enumerate(doc, start=1):
        try:
            # Extract text with better formatting
            text = page.get_text("text").strip()
            
            # Try to extract title from first line
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            heading = lines[0][:100] if lines else f"Page {pno}"
            
            section = Section(
                section_id=f"page-{pno}",
                heading=heading,
                content=text,
                location=f"page:{pno}",
                images=[]
            )
            sections.append(section)

            # Extract images with better error handling
            img_list = page.get_images(full=True)
            for img_idx, img in enumerate(img_list):
                xref = img[0]
                try:
                    pix = fitz.Pixmap(doc, xref)
                    
                    # Skip tiny images (likely decorative)
                    if pix.width < 50 or pix.height < 50:
                        continue
                    
                    image_id = f"{uuid.uuid4().hex}_p{pno}_{img_idx}.png"
                    out_path = os.path.join(IMG_DIR, image_id)
                    
                    if _save_pixmap(pix, out_path):
                        context = text[:500] if text else ""
                        img_obj = Image(
                            image_id=image_id,
                            path=out_path,
                            page=pno,
                            context_text=context
                        )
                        images.append(img_obj)
                        section.images.append(image_id)
                        
                except Exception as e:
                    logger.warning(f"Failed to extract image on page {pno}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error processing page {pno}: {e}")
            continue
    
    doc.close()
    return sections, images

def extract_docx(path: str) -> Tuple[List[Section], List[Image]]:
    """
    Extract structured content from DOCX with improved heading detection.
    """
    try:
        doc = docx.Document(path)
    except Exception as e:
        raise ValueError(f"Cannot open DOCX: {e}")
    
    sections = []
    images = []
    
    current = Section(
        section_id="intro",
        heading="Introduction",
        content="",
        location="docx:0"
    )
    sec_count = 0

    for idx, para in enumerate(doc.paragraphs):
        style_name = getattr(para.style, "name", "").lower()
        text = para.text.strip()
        
        if not text:
            continue
        
        # Detect headings
        if "heading" in style_name or (len(text) < 100 and para.runs and para.runs[0].bold):
            # Save previous section
            if current.content.strip():
                sections.append(current)
            
            sec_count += 1
            current = Section(
                section_id=f"sec-{sec_count}",
                heading=text[:200] if text else f"Section {sec_count}",
                content="",
                location=f"docx:{idx}"
            )
        else:
            current.content += text + "\n\n"

    # Save last section
    if current.content.strip():
        sections.append(current)
    
    # DOCX image extraction could be added here using python-docx-template or docx2python
    
    return sections, images

# ----------------- RAG INDEX -----------------
class RAGIndex:
    """Retrieval-Augmented Generation index with FAISS"""
    
    def __init__(self, embed_model_name: str = EMBED_MODEL):
        global EMBED_MODEL_OBJ
        if EMBED_MODEL_OBJ is None:
            logger.info(f"Loading embedding model: {embed_model_name}")
            EMBED_MODEL_OBJ = SentenceTransformer(embed_model_name)
        self.embed_model = EMBED_MODEL_OBJ
        self.index = None
        self.sections = []
        self.embeddings = None

    def build(self, sections: List[Section]):
        """Build FAISS index from sections"""
        if not sections:
            raise ValueError("No sections to index")
        
        texts = [s.content or "" for s in sections]
        
        # Filter out empty texts
        valid_sections = [s for s, t in zip(sections, texts) if t.strip()]
        valid_texts = [t for t in texts if t.strip()]
        
        if not valid_texts:
            raise ValueError("No valid text content to index")
        
        logger.info(f"Encoding {len(valid_texts)} text chunks...")
        embeddings = self.embed_model.encode(valid_texts, convert_to_numpy=True, show_progress_bar=False)
        embeddings = np.array(embeddings).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner Product for cosine similarity
        self.index.add(embeddings)
        self.sections = valid_sections
        self.embeddings = embeddings
        
        logger.info(f"Built FAISS index: {embeddings.shape[0]} vectors, dim={dim}")

    def search(self, query: str, k: int = 3) -> List[Tuple[Section, float]]:
        """Search for relevant sections, returns (section, score) tuples"""
        if self.index is None:
            raise RuntimeError("Index not built yet")
        
        # Encode and normalize query
        q_emb = self.embed_model.encode([query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(q_emb)
        
        # Search
        scores, indices = self.index.search(q_emb, min(k, len(self.sections)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.sections):
                results.append((self.sections[int(idx)], float(score)))
        
        return results

# ----------------- LLM WRAPPER -----------------
def generate_with_gemini(prompt: str, model: str = GEMINI_MODEL, max_retries: int = 2) -> str:
    """
    Call Gemini with retry logic and better error handling.
    """
    for attempt in range(max_retries):
        try:
            mdl = genai.GenerativeModel(model)
            response = mdl.generate_content(prompt)

            # Handle blocked content
            if hasattr(response, "prompt_feedback"):
                feedback = response.prompt_feedback
                if hasattr(feedback, "block_reason") and feedback.block_reason:
                    return "⚠️ The AI couldn't generate a response due to content policy restrictions."

            if hasattr(response, "text"):
                return response.text.strip()

            # Fallback
            return str(response).strip()

        except Exception as e:
            logger.warning(f"Gemini attempt {attempt + 1} failed: {e}")

            if attempt == max_retries - 1:
                return f"❌ AI service error: {str(e)[:200]}"

            time.sleep(1)

    return "❌ Failed to get AI response after retries."

# ----------------- TELEGRAM HANDLERS -----------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Welcome message with instructions"""
    welcome = """
👋 **Welcome to Document Tutor!**

I help you learn from your documents using AI. Here's what I can do:

📄 **Upload** a PDF or DOCX file
🤖 **Ask questions** about the content
📚 **Teach sections** with `teach N` (section number)
📊 **Get summaries** with `/summary`
🎯 **Take quizzes** with `/quiz`
📋 **List sections** with `/sections`
🔄 **Clear document** with `/clear`

Just send me a document to get started!
"""
    await update.message.reply_text(welcome, parse_mode='Markdown')

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show help information"""
    help_text = """
**Available Commands:**

/start - Show welcome message
/help - Show this help
/sections - List all document sections
/summary - Get document summary
/quiz - Generate a quiz
/clear - Clear current document
/stats - Show document statistics

**Usage Examples:**
• `teach 1` - Teach section 1
• `What is...?` - Ask any question
• `explain X` - Get detailed explanation
"""
    await update.message.reply_text(help_text, parse_mode='Markdown')

async def cmd_sections(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """List all sections in the current document"""
    chat_id = update.effective_chat.id
    
    if chat_id not in USER_STATE:
        await update.message.reply_text("📭 No document loaded. Please upload one first.")
        return
    
    state = USER_STATE[chat_id]
    sections = state.sections
    
    if not sections:
        await update.message.reply_text("No sections found in document.")
        return
    
    msg = f"📑 **{state.document_name}** - {len(sections)} sections:\n\n"
    for i, sec in enumerate(sections, 1):
        preview = sec.content[:80].replace('\n', ' ') if sec.content else "Empty"
        msg += f"{i}. **{sec.heading}**\n   _{preview}..._\n\n"
        
        if len(msg) > 3500:  # Leave room for footer
            msg += f"... and {len(sections) - i} more sections"
            break
    
    await safe_send_text(context.bot, chat_id, msg)

async def cmd_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate document summary"""
    chat_id = update.effective_chat.id
    
    if chat_id not in USER_STATE:
        await update.message.reply_text("📭 No document loaded.")
        return
    
    state = USER_STATE[chat_id]
    await update.message.reply_text("📝 Generating summary...")
    
    # Create condensed version for LLM
    doc_text = "\n\n".join([
        f"## {s.heading}\n{s.content[:1000]}"
        for s in state.sections[:10]  # Limit sections
    ])
    
    prompt = f"""Create a comprehensive summary of this document in 3-4 paragraphs. Include:
1. Main topic and purpose
2. Key concepts and ideas
3. Important takeaways
4. Who would benefit from this

Document:
{doc_text[:10000]}
"""
    
    summary = await run_blocking(generate_with_gemini, prompt)
    await safe_send_text(context.bot, chat_id, f"📊 **Document Summary**\n\n{summary}")

async def cmd_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate a quiz from the document"""
    chat_id = update.effective_chat.id
    
    if chat_id not in USER_STATE:
        await update.message.reply_text("📭 No document loaded.")
        return
    
    state = USER_STATE[chat_id]
    await update.message.reply_text("🎯 Generating quiz...")
    
    # Sample content from various sections
    content_samples = "\n\n".join([
        s.content[:800] for s in state.sections[:5]
    ])
    
    prompt = f"""Create a 5-question multiple choice quiz based on this content. Format each question as:

Q1: [Question]
A) [Option]
B) [Option]
C) [Option]
D) [Option]
Answer: [Letter]

Content:
{content_samples}
"""
    
    quiz = await run_blocking(generate_with_gemini, prompt)
    await safe_send_text(context.bot, chat_id, f"🎯 **Quiz Time!**\n\n{quiz}")

async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clear current document"""
    chat_id = update.effective_chat.id
    
    if chat_id in USER_STATE:
        del USER_STATE[chat_id]
        await update.message.reply_text("🗑️ Document cleared. Upload a new one!")
    else:
        await update.message.reply_text("No document to clear.")

async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show document statistics"""
    chat_id = update.effective_chat.id
    
    if chat_id not in USER_STATE:
        await update.message.reply_text("📭 No document loaded.")
        return
    
    state = USER_STATE[chat_id]
    
    total_chars = sum(len(s.content) for s in state.sections)
    total_words = sum(len(s.content.split()) for s in state.sections)
    
    stats = f"""
📊 **Document Statistics**

📄 Name: {state.document_name}
📑 Sections: {len(state.sections)}
🖼️ Images: {len(state.images)}
📝 Words: {total_words:,}
🔤 Characters: {total_chars:,}
💬 Questions asked: {len(state.conversation_history)}
⏰ Uploaded: {state.upload_time.strftime('%Y-%m-%d %H:%M')}
"""
    
    await update.message.reply_text(stats, parse_mode='Markdown')

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle document upload"""
    chat_id = update.effective_chat.id
    doc = update.message.document
    
    # Check file size
    if doc.file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        await update.message.reply_text(
            f"❌ File too large. Maximum size: {MAX_FILE_SIZE_MB}MB"
        )
        return
    
    # Check file type
    filename = doc.file_name.lower()
    if not (filename.endswith('.pdf') or filename.endswith('.docx')):
        await update.message.reply_text(
            "❌ Unsupported format. Please send PDF or DOCX files."
        )
        return
    
    msg = await update.message.reply_text("⬇️ Downloading...")
    
    # Download file
    try:
        file = await context.bot.get_file(doc.file_id)
        local_name = f"{chat_id}_{uuid.uuid4().hex}_{doc.file_name}"
        local_path = os.path.join(DOWNLOAD_DIR, local_name)
        await file.download_to_drive(local_path)
    except Exception as e:
        logger.exception("Download failed")
        await msg.edit_text(f"❌ Download failed: {e}")
        return
    
    # Extract content
    await msg.edit_text("📖 Extracting content...")
    try:
        if filename.endswith('.pdf'):
            sections, images = await run_blocking(extract_pdf, local_path)
        else:
            sections, images = await run_blocking(extract_docx, local_path)
    except Exception as e:
        logger.exception("Extraction failed")
        await msg.edit_text(f"❌ Extraction failed: {e}")
        return
    
    if not sections:
        await msg.edit_text("❌ No text content found in document.")
        return
    
    # Build RAG index
    await msg.edit_text("🔍 Building search index...")
    rag = RAGIndex()
    try:
        await run_blocking(rag.build, sections)
    except Exception as e:
        logger.exception("RAG build failed")
        await msg.edit_text(f"❌ Failed to build index: {e}")
        return
    
    # Store state
    state = UserState(
        sections=sections,
        images=images,
        rag=rag,
        document_name=doc.file_name
    )
    USER_STATE[chat_id] = state
    
    # Generate overview
    await msg.edit_text("✨ Generating overview...")
    preview = "\n\n".join([
        f"{s.heading}:\n{s.content[:600]}" for s in sections[:5]
    ])
    
    prompt = f"""Analyze this document and provide:
1. A 2-3 sentence TL;DR
2. Main topics covered (bullet points)
3. Who should read this

Document preview:
{preview}
"""
    
    overview = await run_blocking(generate_with_gemini, prompt)
    
    # Send results
    await msg.delete()
    result = f"""
✅ **Document Processed!**

📄 {doc.file_name}
📑 {len(sections)} sections, {len(images)} images

{overview}

💡 Try: `teach 1` or ask me questions!
"""
    await safe_send_text(context.bot, chat_id, result)

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages - teaching requests or questions"""
    chat_id = update.effective_chat.id
    text = (update.message.text or "").strip()
    
    if not text:
        return
    
    if chat_id not in USER_STATE:
        await update.message.reply_text(
            "📭 Please upload a document first using /start"
        )
        return
    
    state = USER_STATE[chat_id]
    
    # Handle "teach N" command
    if text.lower().startswith("teach"):
        try:
            parts = text.split()
            if len(parts) < 2:
                await update.message.reply_text(
                    "Usage: `teach N` where N is section number\nUse /sections to see all sections",
                    parse_mode='Markdown'
                )
                return
            
            idx = int(parts[1]) - 1
            if idx < 0 or idx >= len(state.sections):
                await update.message.reply_text(
                    f"❌ Invalid section. Choose 1-{len(state.sections)}"
                )
                return
            
            section = state.sections[idx]
            await update.message.reply_text("👨‍🏫 Preparing lesson...")
            
            prompt = f"""You are an expert tutor. Teach this section clearly and engagingly:

**Section: {section.heading}**

Content:
{section.content[:3000]}

Instructions:
- Start with a brief introduction
- Break down complex concepts
- Use simple examples and analogies
- Highlight key points
- End with a summary

Be encouraging and clear!"""
            
            response = await run_blocking(generate_with_gemini, prompt)
            state.current_section = idx
            state.conversation_history.append({
                "type": "teach",
                "section": idx,
                "timestamp": datetime.now().isoformat()
            })
            
            await safe_send_text(context.bot, chat_id, f"📚 **Teaching: {section.heading}**\n\n{response}")
            
            # Offer images if available
            if section.images:
                keyboard = [[InlineKeyboardButton("📸 Show images", callback_data=f"img_{idx}")]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"This section has {len(section.images)} image(s).",
                    reply_markup=reply_markup
                )
            
            return
            
        except ValueError:
            await update.message.reply_text("Usage: `teach 1` (section number)", parse_mode='Markdown')
            return
        except Exception as e:
            logger.exception("Teach error")
            await update.message.reply_text(f"❌ Error: {e}")
            return
    
    # Handle as Q&A with RAG
    await update.message.reply_text("🔍 Searching...")
    
    try:
        results = await run_blocking(state.rag.search, text, k=3)
    except Exception as e:
        logger.exception("RAG search failed")
        await update.message.reply_text("❌ Search failed. Try rephrasing your question.")
        return
    
    if not results:
        await update.message.reply_text(
            "🤷 Couldn't find relevant information. Try rephrasing or use /sections to explore."
        )
        return
    
    # Build context from top results
    context_parts = []
    for i, (section, score) in enumerate(results[:3], 1):
        context_parts.append(f"[Source {i}: {section.heading}]\n{section.content[:1500]}")
    
    context_text = "\n\n".join(context_parts)
    
    prompt = f"""Answer the user's question using ONLY the provided context. Be specific and cite which source you're using.

If the context doesn't contain the answer, say "I don't have enough information in the document to answer that."

Context:
{context_text}

Question: {text}

Instructions:
- Be direct and clear
- Cite sources like "According to [Source 1: Title]..."
- If multiple sources agree, mention that
- Keep it concise but complete"""
    
    answer = await run_blocking(generate_with_gemini, prompt)
    
    # Log conversation
    state.conversation_history.append({
        "type": "qa",
        "question": text,
        "timestamp": datetime.now().isoformat()
    })
    
    await safe_send_text(context.bot, chat_id, f"💬 **Answer:**\n\n{answer}")
    
    # Show related images if available
    top_section = results[0][0]
    if top_section.images:
        state.pending_images = top_section.images
        keyboard = [[InlineKeyboardButton("📸 Show related images", callback_data="show_imgs")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"Found {len(top_section.images)} related image(s)",
            reply_markup=reply_markup
        )

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline button callbacks"""
    query = update.callback_query
    await query.answer()
    
    chat_id = query.message.chat_id
    
    if chat_id not in USER_STATE:
        await query.edit_message_text("❌ Session expired. Please upload document again.")
        return
    
    state = USER_STATE[chat_id]
    data = query.data
    
    try:
        if data.startswith("img_"):
            # Show images for specific section
            idx = int(data.split("_")[1])
            if idx < 0 or idx >= len(state.sections):
                await query.edit_message_text("❌ Invalid section")
                return
                
            section = state.sections[idx]
            await query.edit_message_text(f"Sending {len(section.images)} image(s)...")
            
            for img_id in section.images[:5]:  # Limit to 5
                img = next((i for i in state.images if i.image_id == img_id), None)
                if img and os.path.exists(img.path):
                    try:
                        with open(img.path, 'rb') as f:
                            caption = f"📄 Page {img.page}\n{img.context_text[:200]}"
                            await context.bot.send_photo(
                                chat_id=chat_id,
                                photo=f,
                                caption=caption
                            )
                            await asyncio.sleep(0.3)  # Rate limiting
                    except Exception as e:
                        logger.warning(f"Failed to send image {img_id}: {e}")
                        
        elif data == "show_imgs":
            # Show pending images from last Q&A
            if not state.pending_images:
                await query.edit_message_text("No images available")
                return
                
            await query.edit_message_text(f"Sending {len(state.pending_images)} image(s)...")
            
            for img_id in state.pending_images[:5]:
                img = next((i for i in state.images if i.image_id == img_id), None)
                if img and os.path.exists(img.path):
                    try:
                        with open(img.path, 'rb') as f:
                            caption = f"📄 Page {img.page}\n{img.context_text[:200]}"
                            await context.bot.send_photo(
                                chat_id=chat_id,
                                photo=f,
                                caption=caption
                            )
                            await asyncio.sleep(0.3)
                    except Exception as e:
                        logger.warning(f"Failed to send image {img_id}: {e}")
            
            state.pending_images = []
        else:
            await query.edit_message_text("❌ Unknown action")
            
    except Exception as e:
        logger.exception("Callback error")
        await query.edit_message_text(f"❌ Error: {e}")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle errors"""
    logger.error(f"Update {update} caused error {context.error}")
    
    if update and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "❌ An error occurred. Please try again or use /start to restart."
            )
        except Exception:
            pass

def main():
    """Main entry point"""
    logger.info("Starting Document Tutor Bot...")
    
    # Build application
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Register handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("sections", cmd_sections))
    app.add_handler(CommandHandler("summary", cmd_summary))
    app.add_handler(CommandHandler("quiz", cmd_quiz))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("stats", cmd_stats))
    
    # Document handler
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    
    # Text handler (for questions and teaching)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    
    # Callback handler
    app.add_handler(CallbackQueryHandler(handle_callback))
    
    # Error handler
    app.add_error_handler(error_handler)
    
    # Start bot
    logger.info("Bot is running. Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()