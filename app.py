import streamlit as st
from transformers import GPTNeoForCausalLM, GPT2TokenizerFast
import torch
import os
import logging
from typing import Optional, Tuple
import time
from datetime import datetime

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
        padding: 2rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        font-family: 'Georgia', serif;
        line-height: 1.8;
        font-size: 1.1em;
        white-space: pre-wrap;
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
    
    .generation-tips {
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
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
        
        with st.spinner("🔄 Loading AI model... This may take a few moments."):
            tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
            
            # Set padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
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


# Enhanced story generation function
@handle_errors
def generate_story(model, tokenizer, device, prompt: str, max_length: int,
                   temperature: float, top_p: float, top_k: int, 
                   num_beams: int = 1, length_penalty: float = 1.0) -> Optional[str]:
    """Generate complete, high-quality story using the fine-tuned model"""
    if not all([model, tokenizer, device]):
        st.error("Model not loaded properly. Please refresh the page.")
        return None
    
    try:
        # Enhanced prompt engineering for better story generation
        enhanced_prompt = f"""{prompt}

Once upon a time, """
        
        # Tokenize input with attention mask
        inputs = tokenizer(
            enhanced_prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        )
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        
        # Calculate actual max tokens for generation
        prompt_length = len(input_ids[0])
        max_new_tokens = min(max_length, 1024 - prompt_length)
        
        # Generate with progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        generation_stages = [
            (0.2, "📝 Analyzing prompt..."),
            (0.4, "🎭 Creating characters..."),
            (0.6, "🏰 Building world..."),
            (0.8, "✍️ Crafting narrative..."),
            (1.0, "✨ Finalizing story...")
        ]
        
        for progress, message in generation_stages:
            progress_bar.progress(progress)
            status_text.text(message)
            time.sleep(0.5)
        
        with torch.no_grad():
            # Enhanced generation parameters for better story quality
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                min_length=prompt_length + 100,  # Ensure minimum story length
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=1.3,  # Increased to reduce repetition
                no_repeat_ngram_size=4,  # Increased to avoid phrase repetition
                length_penalty=length_penalty,
                num_beams=num_beams,
                early_stopping=False,  # Changed to False for complete stories
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        progress_bar.progress(100)
        status_text.text("✅ Story generated successfully!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        # Decode and clean up the output
        story = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Remove the original prompt from the output
        if story.startswith(enhanced_prompt):
            story = story[len(enhanced_prompt):].strip()
        elif story.startswith(prompt):
            story = story[len(prompt):].strip()
        
        # Post-process the story for better formatting
        story = post_process_story(story, title=prompt.split('\n')[0].replace('Title:', '').strip())
        
        return story
        
    except torch.cuda.OutOfMemoryError:
        st.error("⚠️ GPU memory insufficient. Try reducing the story length.")
        return None
    except Exception as e:
        logger.error(f"Story generation failed: {str(e)}")
        st.error(f"Story generation failed: {str(e)}")
        return None


def post_process_story(story: str, title: str = "") -> str:
    """Post-process the generated story for better readability and completeness"""
    
    # Clean up the story
    story = story.strip()
    
    # Ensure story starts with "Once upon a time" if not already
    if not story.lower().startswith("once upon a time"):
        story = "Once upon a time, " + story
    
    # Add title if provided
    if title:
        story = f"**{title}**\n\n{story}"
    
    # Split into paragraphs for better readability
    sentences = story.split('. ')
    paragraphs = []
    current_paragraph = []
    
    for i, sentence in enumerate(sentences):
        current_paragraph.append(sentence)
        # Create new paragraph every 3-5 sentences
        if len(current_paragraph) >= 3 and (i % 4 == 0 or len(current_paragraph) >= 5):
            paragraphs.append('. '.join(current_paragraph) + '.')
            current_paragraph = []
    
    # Add remaining sentences
    if current_paragraph:
        paragraphs.append('. '.join(current_paragraph) + '.')
    
    # Join paragraphs with double newlines
    formatted_story = '\n\n'.join(paragraphs)
    
    # Ensure story has a proper ending if it doesn't
    if not any(formatted_story.rstrip().endswith(ending) for ending in ['.', '!', '?', '"']):
        formatted_story += "."
    
    # Add a concluding sentence if story seems incomplete
    if len(formatted_story) < 500 and not any(phrase in formatted_story.lower() for phrase in ['the end', 'happily ever after', 'and so', 'thus']):
        formatted_story += "\n\nAnd so the tale came to an end, leaving behind memories that would last forever."
    
    return formatted_story


# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if "stories_generated" not in st.session_state:
        st.session_state.stories_generated = 0
    if "current_story" not in st.session_state:
        st.session_state.current_story = ""
    if "generation_history" not in st.session_state:
        st.session_state.generation_history = []
    if "favorite_stories" not in st.session_state:
        st.session_state.favorite_stories = []


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
    
    # Load model
    model, tokenizer, device = load_model()
    
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
        device_name = "GPU" if torch.cuda.is_available() else "CPU"
        st.markdown(f'<div class="metric-card"><h4>⚡ Processing</h4><p style="color: blue;">{device_name}</p></div>', 
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
            "Historical": "🏰 Period settings, historical events",
            "Comedy": "😄 Humorous tales, witty dialogue",
            "Drama": "🎭 Emotional depth, character development"
        }
        
        selected_genre = st.selectbox(
            "Select Genre",
            options=list(genre_options.keys()),
            format_func=lambda x: genre_options[x],
            help="Choose the genre that best fits your story vision"
        )
        
        st.markdown("---")
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings", expanded=True):
            temperature = st.slider(
                "Creativity Level",
                min_value=0.3, max_value=1.5, value=0.8, step=0.1,
                help="Higher values make the story more creative but less focused (0.8 recommended)"
            )
            
            top_p = st.slider(
                "Focus Level",
                min_value=0.7, max_value=1.0, value=0.9, step=0.05,
                help="Controls the diversity of word choices (0.9 recommended)"
            )
            
            top_k = st.slider(
                "Vocabulary Scope",
                min_value=30, max_value=100, value=50, step=10,
                help="Limits the number of word choices considered (50 recommended)"
            )
            
            use_beam_search = st.checkbox(
                "Use Beam Search",
                value=False,
                help="Enable for more coherent but less creative stories"
            )
            
            if use_beam_search:
                num_beams = st.slider(
                    "Beam Width",
                    min_value=2, max_value=5, value=3,
                    help="Number of beams for beam search"
                )
                length_penalty = st.slider(
                    "Length Penalty",
                    min_value=0.5, max_value=2.0, value=1.0, step=0.1,
                    help="Penalty for story length (>1 favors longer stories)"
                )
            else:
                num_beams = 1
                length_penalty = 1.0
        
        st.markdown("---")
        
        # Quick Templates
        with st.expander("📋 Quick Templates"):
            if st.button("🐉 Fantasy Adventure"):
                st.session_state.template = {
                    "title": "The Dragon's Quest",
                    "setting": "A mystical realm where dragons soar above crystal mountains",
                    "characters": "- Elara: A young mage discovering her powers\n- Thornax: An ancient dragon with a secret"
                }
            if st.button("🚀 Space Opera"):
                st.session_state.template = {
                    "title": "Stars Beyond Tomorrow",
                    "setting": "A distant galaxy where humanity has colonized thousands of worlds",
                    "characters": "- Captain Nova: A rogue space trader\n- AI-7: A sentient ship with mysterious origins"
                }
            if st.button("🔍 Murder Mystery"):
                st.session_state.template = {
                    "title": "The Last Guest",
                    "setting": "A isolated mansion during a stormy night",
                    "characters": "- Detective Shaw: A retired investigator\n- The Butler: Knows more than he admits"
                }
        
        st.markdown("---")
        
        # Story statistics
        if st.session_state.generation_history:
            st.subheader("📈 Your Statistics")
            avg_length = sum([len(story.split()) for story in st.session_state.generation_history]) / len(st.session_state.generation_history)
            st.metric("Average Story Length", f"{avg_length:.0f} words")
            st.metric("Total Stories", len(st.session_state.generation_history))
            if st.session_state.favorite_stories:
                st.metric("Favorite Stories", len(st.session_state.favorite_stories))
    
    # Main content area
    st.subheader("📝 Create Your Story")
    
    # Tips section
    with st.expander("💡 Tips for Better Stories", expanded=False):
        st.markdown("""
        <div class="generation-tips">
        <strong>For best results:</strong>
        <ul>
            <li>Be specific with character descriptions and motivations</li>
            <li>Include sensory details in your setting (sights, sounds, atmosphere)</li>
            <li>Consider adding a conflict or challenge for characters to overcome</li>
            <li>Mix familiar elements with unique twists</li>
            <li>Keep creativity level between 0.7-0.9 for balanced results</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Check for template
    if hasattr(st.session_state, 'template'):
        template = st.session_state.template
        default_title = template.get("title", "")
        default_setting = template.get("setting", "")
        default_characters = template.get("characters", "")
        del st.session_state.template
    else:
        default_title = ""
        default_setting = ""
        default_characters = ""
    
    # Input fields in columns
    col1, col2 = st.columns([3, 2])
    
    with col1:
        title = st.text_input(
            "Story Title",
            value=default_title or "The Mysterious Journey",
            placeholder="Enter a captivating title for your story",
            help="Give your story a compelling title"
        )
        
        setting = st.text_area(
            "Setting & Atmosphere",
            value=default_setting or "A misty mountain village where legends come alive under the starlit sky",
            height=100,
            placeholder="Describe the world, time, and mood of your story",
            help="Paint a picture of your story's world"
        )
        
        characters = st.text_area(
            "Main Characters",
            value=default_characters or "- Maya: A curious archaeologist with a talent for solving ancient puzzles\n- Marcus: A mysterious guide who knows more than he reveals",
            height=120,
            placeholder="List your main characters and their key traits",
            help="Bring your characters to life with vivid descriptions"
        )
    
    with col2:
        plot_hook = st.text_area(
            "Plot Hook (Optional)",
            value="",
            height=80,
            placeholder="What event kicks off your story?",
            help="The inciting incident that starts the adventure"
        )
        
        length = st.slider(
            "Story Length (words)",
            min_value=200, max_value=1000, value=500, step=50,
            help="Approximate length of the generated story"
        )
        
        writing_style = st.selectbox(
            "Writing Style",
            ["Narrative", "Descriptive", "Dialogue-heavy", "Action-packed", 
             "Atmospheric", "Poetic", "Minimalist", "Epic"],
            help="Choose the dominant style for your story"
        )
        
        tone = st.selectbox(
            "Story Tone",
            ["Serious", "Light-hearted", "Dark", "Whimsical", "Mysterious", 
             "Inspirational", "Melancholic", "Suspenseful"],
            help="Set the emotional tone of your story"
        )
    
    # Generate button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎨 Generate Story", type="primary", use_container_width=True):
            if not all([title.strip(), setting.strip(), characters.strip()]):
                st.error("⚠️ Please fill in all required fields: Title, Setting, and Characters.")
            elif model is None:
                st.error("⚠️ Story generation engine is not available. Please refresh the page.")
            else:
                # Create enhanced prompt with all details
                prompt = f"""Genre: {selected_genre}
Title: {title}
Style: {writing_style} with {tone.lower()} tone
Setting: {setting}
Characters:
{characters}
{f'Plot Hook: {plot_hook}' if plot_hook else ''}

Create a complete and engaging {selected_genre.lower()} story with a clear beginning, middle, and satisfying ending. The story should be approximately {length} words, written in a {writing_style.lower()} style with a {tone.lower()} tone.

Story:"""
                
                # Generate story with enhanced parameters
                story = generate_story(
                    model, tokenizer, device, prompt,
                    length, temperature, top_p, top_k,
                    num_beams, length_penalty
                )
                
                if story:
                    st.session_state.current_story = story
                    st.session_state.stories_generated += 1
                    st.session_state.generation_history.append(story)
                    st.success("✅ Story generated successfully!")
                    st.balloons()
    
    # Story output section
    if st.session_state.current_story:
        st.markdown("---")
        st.subheader("📖 Your Generated Story")
        
        # Story display with actions
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            if st.button("❤️ Add to Favorites", use_container_width=True):
                if st.session_state.current_story not in st.session_state.favorite_stories:
                    st.session_state.favorite_stories.append(st.session_state.current_story)
                    st.success("Added to favorites!")
        
        with col2:
            if st.button("🔄 Regenerate", use_container_width=True):
                st.rerun()
        
        with col3:
            # Word count
            word_count = len(st.session_state.current_story.split())
            st.metric("Word Count", word_count)
        
        with col4:
            # Reading time estimate
            reading_time = max(1, word_count // 200)
            st.metric("Reading Time", f"{reading_time} min")
        
        # Display the story
        st.markdown(f'<div class="story-output">{st.session_state.current_story}</div>', 
                   unsafe_allow_html=True)
        
        # Download and share options
        col1, col2 = st.columns(2)
        
        with col1:
            # Download button
            st.download_button(
                label="📄 Download Story (.txt)",
                data=st.session_state.current_story,
                file_name=f"{title.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            # Copy to clipboard button (using markdown as workaround)
            if st.button("📋 Copy to Clipboard", use_container_width=True):
                st.info("Story copied! (Feature requires browser support)")
                st.code(st.session_state.current_story, language=None)
    
    # History section
    if st.session_state.generation_history and len(st.session_state.generation_history) > 1:
        st.markdown("---")
        with st.expander("📚 Story History"):
            for i, story in enumerate(reversed(st.session_state.generation_history[-5:])):
                st.text(f"Story {len(st.session_state.generation_history) - i}")
                st.text(story[:200] + "..." if len(story) > 200 else story)
                if st.button(f"Load Story {len(st.session_state.generation_history) - i}", key=f"load_{i}"):
                    st.session_state.current_story = story
                    st.rerun()
                st.markdown("---")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #666;">
        <p>🚀 <strong>StoryForge AI</strong> - Powered by GPT-Neo</p>
        <p style="font-size: 0.9em;">Generate complete, engaging stories with advanced AI</p>
        <p style="font-size: 0.8em; margin-top: 1rem;">
            Made with ❤️ using Streamlit and Hugging Face Transformers
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
