#!/usr/bin/env python3
"""
OneMetrik Image Generator - Streamlit Web Interface
"""

import os
import streamlit as st
import urllib.request
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from generate import (
    MODELS,
    ASPECT_RATIOS,
    SIZES,
    enhance_prompt,
    get_dimensions,
    get_model_dimension_params,
    generate_image,
)
from supabase_client import save_to_vault, get_vault_images, delete_from_vault

# Brand colors
NAVY = "#2D2A5F"
GOLD = "#E8A838"

# Rate limiting
MAX_GENERATIONS = 20

# Path to assets
ASSETS_DIR = Path(__file__).parent.parent / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"


def get_secret(key: str, fallback_env: bool = True) -> str | None:
    """
    Get a secret from st.secrets (Streamlit Cloud) or .env (local dev).

    Args:
        key: The secret key to retrieve
        fallback_env: Whether to fall back to environment variables

    Returns:
        The secret value or None if not found
    """
    # Try st.secrets first (Streamlit Cloud)
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    # Fall back to environment variable
    if fallback_env:
        load_dotenv()
        return os.getenv(key)

    return None


def check_password() -> bool:
    """
    Display login form and verify password.

    Returns:
        True if authenticated, False otherwise
    """
    # Already authenticated
    if st.session_state.get("authenticated", False):
        return True

    # Get the password from secrets
    app_password = get_secret("APP_PASSWORD")

    # If no password configured, allow access (local dev without password)
    if not app_password:
        st.session_state.authenticated = True
        st.session_state.generation_count = 0
        return True

    # Show login form
    st.markdown(f"""
        <style>
        .login-container {{
            max-width: 400px;
            margin: 2rem auto;
            padding: 2rem;
            border: 1px solid #eee;
            border-radius: 8px;
        }}
        </style>
    """, unsafe_allow_html=True)

    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.subheader("Login")

        password = st.text_input("Password", type="password", key="password_input")
        login_clicked = st.button("Login", type="primary", width="stretch")

        if login_clicked:
            if password == app_password:
                st.session_state.authenticated = True
                st.session_state.generation_count = 0
                st.rerun()
            else:
                st.error("Incorrect password")

    return False


def format_model_option(name: str, info: dict) -> str:
    """Format model name with cost for dropdown display."""
    return f"{name} (${info['cost']:.3f}/image) - {info['description']}"


def get_model_from_option(option: str) -> str:
    """Extract model name from formatted dropdown option."""
    return option.split(" (")[0]


def download_image_to_bytes(url: str) -> bytes:
    """Download image from URL and return as bytes."""
    with urllib.request.urlopen(url) as response:
        return response.read()


def truncate_text(text: str, max_length: int = 50) -> str:
    """Truncate text to max_length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length].rstrip() + "..."


def render_vault_page():
    """Render the Vault page showing saved images."""
    # Fetch vault images
    result = get_vault_images()

    if not result["success"]:
        st.error(f"Failed to load Vault: {result.get('error', 'Unknown error')}")
        st.info("Make sure Supabase is configured correctly in your environment.")
        return

    images = result["data"]

    # Header with count
    st.markdown(
        f'## 🗄️ Your Vault <span class="vault-count">({len(images)} images)</span>',
        unsafe_allow_html=True
    )

    # Empty state
    if not images:
        st.markdown(
            '<div class="vault-empty">'
            '<h3>Your vault is empty</h3>'
            '<p>Generate some images and save your favorites!</p>'
            '</div>',
            unsafe_allow_html=True
        )
        return

    # Display images in 3-column grid
    cols = st.columns(3)

    for idx, image in enumerate(images):
        col = cols[idx % 3]

        with col:
            st.markdown('<div class="vault-card">', unsafe_allow_html=True)

            # Image thumbnail (clickable via expander)
            with st.expander("🖼️ View full size", expanded=False):
                st.image(image["image_url"], use_container_width=True)

            # Thumbnail preview
            st.image(image["image_url"], use_container_width=True)

            # Prompt with tooltip
            full_prompt = image["prompt"]
            truncated = truncate_text(full_prompt, 50)
            st.markdown(
                f'<p class="vault-prompt" title="{full_prompt}">{truncated}</p>',
                unsafe_allow_html=True
            )

            # Metadata
            created_date = image["created_at"][:10] if image.get("created_at") else "Unknown"
            st.markdown(
                f'<p class="vault-meta">'
                f'<strong>{image["model"]}</strong> · ${float(image["cost"]):.3f} · {created_date}'
                f'</p>',
                unsafe_allow_html=True
            )

            # Action buttons
            btn_col1, btn_col2 = st.columns(2)

            with btn_col1:
                # Download button - fetch image data
                try:
                    image_bytes = download_image_to_bytes(image["image_url"])
                    st.download_button(
                        label="💾",
                        data=image_bytes,
                        file_name=f"vault_{image['id'][:8]}.webp",
                        mime="image/webp",
                        key=f"download_{image['id']}",
                        use_container_width=True
                    )
                except Exception:
                    st.button("💾", disabled=True, key=f"download_{image['id']}")

            with btn_col2:
                # Delete button with confirmation
                delete_key = f"delete_{image['id']}"
                confirm_key = f"confirm_{image['id']}"

                if st.session_state.get(confirm_key, False):
                    if st.button("✓ Confirm", key=delete_key, type="primary", use_container_width=True):
                        delete_result = delete_from_vault(image["id"])
                        if delete_result["success"]:
                            st.session_state[confirm_key] = False
                            st.rerun()
                        else:
                            st.error("Delete failed")
                else:
                    if st.button("🗑️", key=delete_key, use_container_width=True):
                        st.session_state[confirm_key] = True
                        st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)


def main():
    # Page config with favicon
    page_icon = str(LOGO_PATH) if LOGO_PATH.exists() else "🎨"
    st.set_page_config(
        page_title="OneMetrik Image Generator",
        page_icon=page_icon,
        layout="centered"
    )

    # Custom CSS for brand styling
    st.markdown(f"""
        <style>
        /* Header styling */
        h1, h2, h3 {{
            color: {NAVY} !important;
        }}

        /* Primary button styling (Generate button) */
        .stButton > button[kind="primary"] {{
            background-color: {GOLD} !important;
            border-color: {GOLD} !important;
            color: {NAVY} !important;
            font-weight: 600;
        }}
        .stButton > button[kind="primary"]:hover {{
            background-color: #d4962f !important;
            border-color: #d4962f !important;
        }}

        /* Footer styling */
        .footer {{
            text-align: center;
            color: #888;
            font-size: 0.85rem;
            margin-top: 3rem;
            padding: 1rem 0;
            border-top: 1px solid #eee;
        }}

        /* Rate limit badge */
        .rate-limit {{
            text-align: right;
            font-size: 0.85rem;
            color: #666;
        }}

        /* Vault card styling */
        .vault-card {{
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 0.75rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            background: white;
        }}
        .vault-card:hover {{
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .vault-prompt {{
            font-size: 0.9rem;
            color: #333;
            margin: 0.5rem 0;
            line-height: 1.4;
        }}
        .vault-meta {{
            font-size: 0.8rem;
            color: #666;
        }}
        .vault-empty {{
            text-align: center;
            padding: 3rem;
            color: #888;
        }}
        .vault-count {{
            color: {GOLD};
            font-weight: 600;
        }}

        /* Discard button styling */
        .stButton > button[kind="secondary"] {{
            border-color: #ccc !important;
            color: #666 !important;
        }}
        </style>
    """, unsafe_allow_html=True)

    # Logo (centered)
    if LOGO_PATH.exists():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(str(LOGO_PATH), width=200)

    st.title("OneMetrik Image Generator")
    st.caption("Generate AI images using Replicate API")

    # Check password before proceeding
    if not check_password():
        st.markdown('<div class="footer">Powered by OneMetrik</div>', unsafe_allow_html=True)
        st.stop()

    # Sidebar navigation
    with st.sidebar:
        st.markdown(f"### Navigation")
        page = st.radio(
            "Go to",
            ["🎨 Generate", "🗄️ Vault"],
            label_visibility="collapsed"
        )

    # Route to appropriate page
    if page == "🗄️ Vault":
        render_vault_page()
        st.markdown('<div class="footer">Powered by OneMetrik</div>', unsafe_allow_html=True)
        st.stop()

    # --- Generate Page ---
    # Check API token (st.secrets first, then .env)
    api_token = get_secret("REPLICATE_API_TOKEN")
    if not api_token:
        st.error("REPLICATE_API_TOKEN not found. Please add it to secrets or .env file.")
        st.stop()

    # Set token in environment for the generate module
    os.environ["REPLICATE_API_TOKEN"] = api_token

    # Initialize generation count if not set
    if "generation_count" not in st.session_state:
        st.session_state.generation_count = 0

    # Show rate limit status
    remaining = MAX_GENERATIONS - st.session_state.generation_count
    st.markdown(
        f'<div class="rate-limit">Generations remaining: {remaining}/{MAX_GENERATIONS}</div>',
        unsafe_allow_html=True
    )

    # Prompt input
    prompt = st.text_area(
        "Prompt",
        placeholder="Describe the image you want to generate...",
        height=100
    )

    # Settings in columns
    col1, col2 = st.columns(2)

    with col1:
        # Model selection
        model_options = [format_model_option(name, info) for name, info in MODELS.items()]
        selected_model_option = st.selectbox("Model", model_options)
        model_name = get_model_from_option(selected_model_option)

        # Aspect ratio
        aspect_ratio = st.selectbox(
            "Aspect Ratio",
            options=list(ASPECT_RATIOS.keys()),
            format_func=lambda x: f"{x} - {ASPECT_RATIOS[x]['description']}"
        )

    with col2:
        # Size
        size = st.selectbox(
            "Size",
            options=list(SIZES.keys()),
            index=2,  # Default to "large"
            format_func=lambda x: f"{x} ({SIZES[x]['base_px']}px)"
        )

        # Enhance prompt checkbox
        enhance = st.checkbox("Enhance prompt", help="Add professional photography terms")

    # Calculate and display cost estimate
    width, height = get_dimensions(aspect_ratio, size)
    cost = MODELS[model_name]["cost"]

    st.divider()

    # Cost and dimension info
    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.metric("Dimensions", f"{width} x {height}")
    with info_col2:
        st.metric("Estimated Cost", f"${cost:.3f}")

    # Generate button
    generate_clicked = st.button("Generate Image", type="primary", width="stretch")

    if generate_clicked:
        if not prompt.strip():
            st.error("Please enter a prompt")
            st.stop()

        # Check rate limit
        if st.session_state.generation_count >= MAX_GENERATIONS:
            st.error("Rate limit reached. Please log in again to reset.")
            st.stop()

        # Prepare the prompt
        if enhance:
            final_prompt = enhance_prompt(prompt)
        else:
            final_prompt = prompt

        # Get dimension params for API
        dimension_params = get_model_dimension_params(
            model_name, width, height, aspect_ratio, size
        )

        # Generate with spinner
        with st.spinner(f"Generating {width}x{height} image with {model_name}..."):
            try:
                image_url = generate_image(
                    final_prompt,
                    model_name,
                    dimension_params,
                    width,
                    height
                )

                # Download image
                image_bytes = download_image_to_bytes(image_url)

                # Store in session state
                st.session_state.generated_image = image_bytes
                st.session_state.image_metadata = {
                    "prompt": prompt,
                    "enhanced_prompt": final_prompt if enhance else None,
                    "model": model_name,
                    "dimensions": f"{width}x{height}",
                    "aspect_ratio": aspect_ratio,
                    "size": size,
                    "cost": cost,
                    "timestamp": datetime.now().isoformat()
                }

                # Increment generation count
                st.session_state.generation_count += 1
                st.rerun()

            except Exception as e:
                st.error(f"Generation failed: {str(e)}")
                st.stop()

    # Display results if we have a generated image
    if "generated_image" in st.session_state:
        st.divider()
        st.subheader("Generated Image")

        # Display image
        st.image(st.session_state.generated_image, use_container_width=True)

        # Metadata
        meta = st.session_state.image_metadata
        st.markdown(f"**Prompt:** {meta['prompt']}")
        if meta.get("enhanced_prompt"):
            st.markdown(f"**Enhanced:** {meta['enhanced_prompt']}")

        detail_col1, detail_col2, detail_col3 = st.columns(3)
        with detail_col1:
            st.markdown(f"**Model:** {meta['model']}")
        with detail_col2:
            st.markdown(f"**Size:** {meta['dimensions']}")
        with detail_col3:
            st.markdown(f"**Cost:** ${meta['cost']:.3f}")

        st.divider()

        # Action buttons: Add to Vault, Discard, Download
        btn_col1, btn_col2, btn_col3 = st.columns(3)

        with btn_col1:
            if st.button("⭐ Add to Vault", type="primary", use_container_width=True):
                with st.spinner("Saving to Vault..."):
                    result = save_to_vault(
                        image_data=st.session_state.generated_image,
                        prompt=meta["prompt"],
                        model=meta["model"],
                        cost=meta["cost"]
                    )
                    if result["success"]:
                        del st.session_state.generated_image
                        del st.session_state.image_metadata
                        st.success("Added to Vault!")
                        st.rerun()
                    else:
                        st.error(f"Failed to save: {result.get('error', 'Unknown error')}")

        with btn_col2:
            if st.button("🗑️ Discard", use_container_width=True):
                del st.session_state.generated_image
                del st.session_state.image_metadata
                st.info("Discarded - ready for next prompt")
                st.rerun()

        with btn_col3:
            st.download_button(
                label="💾 Download",
                data=st.session_state.generated_image,
                file_name=f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.webp",
                mime="image/webp",
                use_container_width=True
            )

    # Footer
    st.markdown('<div class="footer">Powered by OneMetrik</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
