# streamlit_app.py

import streamlit as st
from transformers import GPTNeoForCausalLM, GPT2TokenizerFast
import torch
from ollama import Client
import os
import logging
from typing import Optional, Tuple
import time
from datetime import datetime
from transformers import GPTNeoForCausalLM, GPT2TokenizerFast




# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="StoryForge AI - Professional Story Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/storyforge-ai',
        'Report a bug': "https://github.com/yourusername/storyforge-ai/issues",
        'About': "StoryForge AI - Create captivating stories with the power of AI"
    }
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }

    .story-output {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        font-family: 'Georgia', serif;
        line-height: 1.6;
    }

    .warning-banner {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }

    .success-banner {
        background: #d1edff;
        border: 1px solid #74b9ff;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: #0984e3;
    }

    .sidebar .stSelectbox label, .sidebar .stSlider label {
        font-weight: 600;
        color: #2d3436;
    }

    .stButton > button {
        width: 100%;
        border-radius: 25px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)


# Error handling decorator
def handle_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            st.error(f"An error occurred: {str(e)}")
            return None

    return wrapper


# Model loading with enhanced error handling
@st.cache_resource(show_spinner=False)
@handle_errors
def load_model() -> Optional[Tuple]:
    """Load the fine-tuned GPT-Neo model and tokenizer"""
    try:
        model_path = "Shashwath45/gptneo-story-model"

        tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
        model = GPTNeoForCausalLM.from_pretrained(model_path)

        with st.spinner("🔄 Loading AI models... This may take a few moments."):
            tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
            model = GPTNeoForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()  # Set to evaluation mode

            logger.info(f"Model loaded successfully on {device}")
            return model, tokenizer, device

    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        st.error("⚠️ Failed to load the AI model. Please check if the model files are available.")
        return None, None, None


# Initialize Ollama client with error handling
@handle_errors
def init_ollama_client():
    """Initialize Ollama client with configuration"""
    try:
        client = Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        return client
    except Exception as e:
        logger.warning(f"Ollama client initialization failed: {str(e)}")
        return None


# Story generation function
@handle_errors
def generate_story(model, tokenizer, device, prompt: str, max_length: int,
                   temperature: float, top_p: float, top_k: int) -> Optional[str]:
    """Generate story using the fine-tuned model"""
    if not all([model, tokenizer, device]):
        st.error("Model not loaded properly. Please refresh the page.")
        return None

    try:
        # Tokenize input
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)

        # Generate with progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=min(max_length + len(input_ids[0]), 1024),  # Prevent excessive length
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                early_stopping=True
            )

        progress_bar.progress(100)
        status_text.text("✅ Story generated successfully!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()

        # Decode and clean up the output
        story = tokenizer.decode(output[0], skip_special_tokens=True)

        # Remove the original prompt from the output
        if story.startswith(prompt):
            story = story[len(prompt):].strip()

        return story

    except torch.cuda.OutOfMemoryError:
        st.error("⚠️ GPU memory insufficient. Try reducing the story length.")
        return None
    except Exception as e:
        logger.error(f"Story generation failed: {str(e)}")
        st.error(f"Story generation failed: {str(e)}")
        return None


# Story refinement function
@handle_errors
def refine_story(ollama_client, story: str, style_instructions: str = "") -> Optional[str]:
    """Refine story using Ollama"""
    if not ollama_client:
        st.warning("⚠️ Story refinement service is not available. The original story will be used.")
        return story

    try:
        system_prompt = f"""You are a professional fiction editor and creative writing expert. 
        Your task is to polish and enhance the given story while preserving its core narrative and tone.

        Guidelines:
        - Improve prose quality and flow
        - Enhance character development and dialogue
        - Strengthen scene descriptions and atmosphere
        - Fix any grammatical or stylistic issues
        - Maintain the original story's length and structure
        - Keep the genre and mood consistent
        {style_instructions}

        Return only the refined story without any additional commentary."""

        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("🎨 Refining your story with AI...")

        response = ollama_client.chat(
            model=os.getenv("OLLAMA_MODEL", "mistral"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please refine this story:\n\n{story}"}
            ],
            options={
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40
            }
        )

        progress_bar.progress(100)
        status_text.text("✅ Story refined successfully!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()

        return response["message"]["content"]

    except Exception as e:
        logger.error(f"Story refinement failed: {str(e)}")
        st.warning(f"⚠️ Story refinement failed: {str(e)}. Using original story.")
        return story


# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if "stories_generated" not in st.session_state:
        st.session_state.stories_generated = 0
    if "current_story" not in st.session_state:
        st.session_state.current_story = ""
    if "refined_story" not in st.session_state:
        st.session_state.refined_story = ""
    if "generation_history" not in st.session_state:
        st.session_state.generation_history = []


# Main application
def main():
    init_session_state()

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📚 StoryForge AI</h1>
        <p>Professional AI-Powered Story Generator</p>
        <p style="font-size: 0.9em; opacity: 0.9;">Create captivating stories with advanced AI technology</p>
    </div>
    """, unsafe_allow_html=True)

    # Load models
    model, tokenizer, device = load_model()
    ollama_client = init_ollama_client()

    # Model status indicator
    col1, col2, col3 = st.columns(3)
    with col1:
        if model is not None:
            st.markdown('<div class="metric-card"><h4>🤖 Story Engine</h4><p style="color: green;">✅ Ready</p></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card"><h4>🤖 Story Engine</h4><p style="color: red;">❌ Offline</p></div>',
                        unsafe_allow_html=True)

    with col2:
        if ollama_client is not None:
            st.markdown('<div class="metric-card"><h4>🎨 Refinement</h4><p style="color: green;">✅ Available</p></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card"><h4>🎨 Refinement</h4><p style="color: orange;">⚠️ Limited</p></div>',
                        unsafe_allow_html=True)

    with col3:
        st.markdown(
            f'<div class="metric-card"><h4>📊 Stories Created</h4><p style="color: blue;">{st.session_state.stories_generated}</p></div>',
            unsafe_allow_html=True)

    # Sidebar configuration
    with st.sidebar:
        st.header("🎯 Story Configuration")

        # Genre selection with descriptions
        genre_options = {
            "Fantasy": "🐉 Magical worlds, mythical creatures, epic quests",
            "Sci-Fi": "🚀 Future technology, space exploration, alien worlds",
            "Mystery": "🔍 Puzzles, investigations, suspenseful revelations",
            "Horror": "👻 Supernatural terror, psychological thrills",
            "Adventure": "⚔️ Action-packed journeys, daring exploits",
            "Romance": "💕 Love stories, emotional connections",
            "Thriller": "⚡ High-stakes tension, fast-paced action",
            "Historical": "🏰 Period settings, historical events"
        }

        selected_genre = st.selectbox(
            "Select Genre",
            options=list(genre_options.keys()),
            format_func=lambda x: genre_options[x],
            help="Choose the genre that best fits your story vision"
        )

        st.markdown("---")

        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            temperature = st.slider(
                "Creativity Level",
                min_value=0.1, max_value=2.0, value=1.0, step=0.1,
                help="Higher values make the story more creative but less focused"
            )

            top_p = st.slider(
                "Focus Level",
                min_value=0.1, max_value=1.0, value=0.95, step=0.05,
                help="Controls the diversity of word choices"
            )

            top_k = st.slider(
                "Vocabulary Scope",
                min_value=10, max_value=100, value=50, step=10,
                help="Limits the number of word choices considered"
            )

        st.markdown("---")

        # Story statistics
        if st.session_state.generation_history:
            st.subheader("📈 Your Statistics")
            avg_length = sum([len(story.split()) for story in st.session_state.generation_history]) / len(
                st.session_state.generation_history)
            st.metric("Average Story Length", f"{avg_length:.0f} words")
            st.metric("Total Stories", len(st.session_state.generation_history))

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📝 Story Details")

        # Input fields
        title = st.text_input(
            "Story Title",
            value="The Lost Kingdom",
            help="Give your story a compelling title"
        )

        setting = st.text_area(
            "Setting & Atmosphere",
            value="A dark forest under a blood-red moon, where ancient magic still lingers in the shadows",
            height=100,
            help="Describe the world, time, and mood of your story"
        )

        characters = st.text_area(
            "Main Characters",
            value="- Aria: A brave hunter with a mysterious past\n- Kael: A cursed prince seeking redemption\n- Elder Thorne: An wise but secretive village elder",
            height=120,
            help="Describe your main characters and their key traits"
        )

        # Story length and additional options
        col_len, col_style = st.columns(2)
        with col_len:
            length = st.slider(
                "Story Length (words)",
                min_value=100, max_value=800, value=400, step=50,
                help="Approximate length of the generated story"
            )

        with col_style:
            writing_style = st.selectbox(
                "Writing Style",
                ["Narrative", "Descriptive", "Dialogue-heavy", "Action-packed", "Atmospheric"],
                help="Choose the dominant style for your story"
            )

    with col2:
        st.subheader("🎬 Story Actions")

        # Generate button
        if st.button("🎨 Generate Story", type="primary", use_container_width=True):
            if not all([title.strip(), setting.strip(), characters.strip()]):
                st.error("⚠️ Please fill in all required fields: Title, Setting, and Characters.")
            elif model is None:
                st.error("⚠️ Story generation engine is not available. Please try again later.")
            else:
                # Create the prompt
                prompt = f"""Genre: {selected_genre}
Title: {title}
Setting: {setting}
Characters:
{characters}
Writing Style: {writing_style}

Story:
"""

                # Generate story
                with st.spinner("🔮 Crafting your story... This may take a moment."):
                    story = generate_story(
                        model, tokenizer, device, prompt,
                        length, temperature, top_p, top_k
                    )

                if story:
                    st.session_state.current_story = story
                    st.session_state.stories_generated += 1
                    st.session_state.generation_history.append(story)
                    st.success("✅ Story generated successfully!")

        # Refine button
        if st.button("🎩 Refine Story", use_container_width=True):
            if not st.session_state.current_story:
                st.warning("⚠️ Please generate a story first before refining.")
            else:
                style_instructions = f"\n- Writing style should be: {writing_style}\n- Genre: {selected_genre}"
                refined = refine_story(ollama_client, st.session_state.current_story, style_instructions)

                if refined:
                    st.session_state.refined_story = refined
                    st.success("✅ Story refined successfully!")

        # Clear button
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.current_story = ""
            st.session_state.refined_story = ""
            st.success("✅ Stories cleared!")

    # Story output section
    if st.session_state.current_story or st.session_state.refined_story:
        st.markdown("---")
        st.subheader("📖 Your Story")

        # Tabs for original and refined versions
        if st.session_state.refined_story:
            tab1, tab2 = st.tabs(["🎨 Refined Version", "📝 Original Version"])

            with tab1:
                st.markdown(f'<div class="story-output">{st.session_state.refined_story}</div>', unsafe_allow_html=True)

                # Download button for refined story
                st.download_button(
                    label="📄 Download Refined Story",
                    data=st.session_state.refined_story,
                    file_name=f"{title.replace(' ', '_').lower()}_refined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            with tab2:
                st.markdown(f'<div class="story-output">{st.session_state.current_story}</div>', unsafe_allow_html=True)

                # Download button for original story
                st.download_button(
                    label="📄 Download Original Story",
                    data=st.session_state.current_story,
                    file_name=f"{title.replace(' ', '_').lower()}_original_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        else:
            st.markdown(f'<div class="story-output">{st.session_state.current_story}</div>', unsafe_allow_html=True)

            # Download button
            st.download_button(
                label="📄 Download Story",
                data=st.session_state.current_story,
                file_name=f"{title.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #666;">
        <p>🚀 <strong>StoryForge AI</strong> - Powered by Advanced Language Models</p>
        <p style="font-size: 0.9em;">Create, refine, and download professional stories with AI assistance</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
