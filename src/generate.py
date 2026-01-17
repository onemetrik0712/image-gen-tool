#!/usr/bin/env python3
"""
OneMetrik Image Generator - Generate ad creative variations using Replicate API
"""

import os
import sys
import argparse
import json
import time
import urllib.request
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv


# Model registry with Replicate IDs and costs
MODELS = {
    "flux-schnell": {
        "id": "black-forest-labs/flux-schnell",
        "cost": 0.003,
        "description": "Fast, high-quality images (default)"
    },
    "flux-pro": {
        "id": "black-forest-labs/flux-1.1-pro",
        "cost": 0.05,
        "description": "Best quality, slower generation"
    },
    "imagen-4": {
        "id": "google/imagen-4",
        "cost": 0.03,
        "description": "Google Imagen 4"
    },
    "seedream": {
        "id": "bytedance/seedream-4",
        "cost": 0.02,
        "description": "ByteDance Seedream 4"
    }
}


# Aspect ratio presets with dimensions at large (1024px base) size
ASPECT_RATIOS = {
    "1:1": {"width": 1024, "height": 1024, "description": "Square (default)"},
    "16:9": {"width": 1344, "height": 768, "description": "Landscape/widescreen"},
    "9:16": {"width": 768, "height": 1344, "description": "Portrait/stories"},
    "4:3": {"width": 1152, "height": 864, "description": "Standard landscape"},
    "3:4": {"width": 864, "height": 1152, "description": "Standard portrait"}
}

# Size multipliers (relative to large = 1.0)
SIZES = {
    "small": {"multiplier": 0.5, "base_px": 512, "description": "512px base"},
    "medium": {"multiplier": 0.75, "base_px": 768, "description": "768px base"},
    "large": {"multiplier": 1.0, "base_px": 1024, "description": "1024px base (default)"},
    "xl": {"multiplier": 1.3125, "base_px": 1344, "description": "1344px base"}
}


def list_models() -> None:
    """Display available models in a formatted table."""
    print("\nAvailable Models:")
    print("-" * 70)
    print(f"{'Model':<20} {'Cost/Image':<15} {'Description':<35}")
    print("-" * 70)

    for name, info in MODELS.items():
        cost_str = f"${info['cost']:.3f}"
        print(f"{name:<20} {cost_str:<15} {info['description']:<35}")

    print("-" * 70)
    print("\nUsage: python src/generate.py \"prompt\" --model flux-schnell\n")
    sys.exit(0)


def list_sizes() -> None:
    """Display available aspect ratios and sizes."""
    print("\nAspect Ratios:")
    print("-" * 55)
    print(f"{'Ratio':<10} {'Dimensions':<15} {'Description':<30}")
    print("-" * 55)
    for ratio, info in ASPECT_RATIOS.items():
        dims = f"{info['width']}x{info['height']}"
        print(f"{ratio:<10} {dims:<15} {info['description']:<30}")

    print("\n\nSizes:")
    print("-" * 45)
    print(f"{'Size':<10} {'Base':<10} {'Description':<25}")
    print("-" * 45)
    for name, info in SIZES.items():
        base = f"{info['base_px']}px"
        print(f"{name:<10} {base:<10} {info['description']:<25}")

    print("\n\nExamples:")
    print("  python src/generate.py \"prompt\" --aspect-ratio 16:9")
    print("  python src/generate.py \"prompt\" --size xl")
    print("  python src/generate.py \"prompt\" --aspect-ratio 9:16 --size medium\n")
    sys.exit(0)


def get_dimensions(aspect_ratio: str, size: str) -> tuple[int, int]:
    """
    Calculate final dimensions from aspect ratio and size.

    Args:
        aspect_ratio: Aspect ratio key (e.g., "16:9")
        size: Size key (e.g., "large")

    Returns:
        Tuple of (width, height)
    """
    ratio_info = ASPECT_RATIOS[aspect_ratio]
    size_info = SIZES[size]

    base_width = ratio_info["width"]
    base_height = ratio_info["height"]
    multiplier = size_info["multiplier"]

    # Round to nearest 64 (many models require dimensions divisible by 64)
    width = round(base_width * multiplier / 64) * 64
    height = round(base_height * multiplier / 64) * 64

    return width, height


def get_model_dimension_params(
    model_name: str,
    width: int,
    height: int,
    aspect_ratio: str,
    size: str
) -> dict[str, Any]:
    """
    Get model-specific dimension parameters for the API call.

    Args:
        model_name: Name of the model
        width: Image width
        height: Image height
        aspect_ratio: Aspect ratio string (e.g., "16:9")
        size: Size preset (e.g., "large")

    Returns:
        Dictionary of parameters to pass to the model
    """
    # Map size presets to megapixels for flux models
    size_to_megapixels = {
        "small": "0.25",
        "medium": "1",
        "large": "1",
        "xl": "1"
    }

    if model_name in ["flux-schnell", "flux-pro"]:
        # Flux models use aspect_ratio + megapixels, not width/height
        return {
            "aspect_ratio": aspect_ratio,
            "megapixels": size_to_megapixels.get(size, "1")
        }
    elif model_name == "imagen-4":
        return {"aspect_ratio": aspect_ratio}
    elif model_name == "seedream":
        return {"width": width, "height": height, "aspect_ratio": aspect_ratio}
    else:
        return {"width": width, "height": height}


def create_prompt_slug(prompt: str) -> str:
    """
    Create a URL-friendly slug from a prompt.

    Args:
        prompt: The original prompt

    Returns:
        Slug with first 30 chars, lowercase, hyphens for spaces
    """
    slug = prompt[:30].lower()
    slug = slug.replace(" ", "-").replace("/", "-").replace("_", "-")
    # Remove any characters that aren't alphanumeric or hyphens
    slug = "".join(c for c in slug if c.isalnum() or c == "-")
    # Remove multiple consecutive hyphens
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


def enhance_prompt(prompt: str) -> str:
    """
    Enhance a basic prompt with professional photography terms.

    Args:
        prompt: The original user prompt

    Returns:
        Enhanced prompt with photography terms
    """
    enhancements = [
        "professional photography",
        "high quality",
        "well lit",
        "sharp focus",
        "detailed",
        "8k resolution"
    ]
    return f"{prompt}, {', '.join(enhancements)}"


def load_api_token() -> str:
    """
    Load the Replicate API token from environment variables.

    Returns:
        The API token string

    Raises:
        SystemExit: If token is not found
    """
    load_dotenv()
    token = os.getenv("REPLICATE_API_TOKEN")

    if not token:
        print("Error: REPLICATE_API_TOKEN not found in .env file")
        print("Please copy .env.example to .env and add your API token")
        sys.exit(1)

    return token


def extract_image_url(output: Any) -> str:
    """
    Extract image URL from different model response formats.

    Args:
        output: The output from replicate.run()

    Returns:
        The image URL string
    """
    # Handle list of URLs
    if isinstance(output, list):
        if len(output) > 0:
            # Could be a list of strings or objects
            first_item = output[0]
            if isinstance(first_item, str):
                return first_item
            elif hasattr(first_item, 'url'):
                return first_item.url
            else:
                return str(first_item)

    # Handle object with url attribute
    if hasattr(output, 'url'):
        return output.url

    # Handle string URL
    if isinstance(output, str):
        return output

    # Fallback
    return str(output)


def generate_image(
    prompt: str,
    model_name: str,
    dimension_params: dict[str, Any],
    width: int,
    height: int,
    current: Optional[int] = None,
    total: Optional[int] = None
) -> str:
    """
    Generate an image using Replicate API with retry logic.

    Args:
        prompt: The text prompt for image generation
        model_name: Name of the model to use (from MODELS dict)
        dimension_params: Model-specific dimension parameters
        width: Target image width (for display)
        height: Target image height (for display)
        current: Current image number (for progress display)
        total: Total number of images (for progress display)

    Returns:
        URL of the generated image

    Raises:
        Exception: If API call fails after retries
    """
    # Import replicate here to avoid Python 3.14 compatibility issues at module load
    try:
        import replicate
    except Exception as e:
        print(f"Error: Failed to import replicate SDK: {e}")
        print("\nThe replicate package is not compatible with Python 3.14.")
        print("Please use Python 3.13 or earlier, or wait for an updated replicate package.")
        sys.exit(1)

    if current and total:
        print(f"Generating {width}x{height} image {current} of {total}...")
    else:
        print(f"Generating {width}x{height} image...")

    model_info = MODELS[model_name]
    model_id = model_info["id"]

    max_retries = 3
    retry_delay = 10

    # Build input params
    input_params = {"prompt": prompt}
    input_params.update(dimension_params)

    for attempt in range(max_retries):
        try:
            output = replicate.run(
                model_id,
                input=input_params
            )

            # Extract URL from response
            return extract_image_url(output)

        except Exception as e:
            error_str = str(e)

            # Check if it's a rate limit error (429)
            if "429" in error_str or "rate limit" in error_str.lower() or "throttled" in error_str.lower():
                if attempt < max_retries - 1:
                    print(f"Rate limited, waiting {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"Error: Rate limit exceeded after {max_retries} attempts")
                    raise
            else:
                # Not a rate limit error, raise immediately
                print(f"Error generating image: {e}")
                raise

    raise Exception("Failed to generate image after maximum retries")


def save_metadata(
    output_folder: Path,
    original_prompt: str,
    enhanced_prompt: Optional[str],
    num_variations: int,
    enhancement_enabled: bool,
    model_name: str,
    total_cost: float,
    width: int,
    height: int,
    aspect_ratio: str,
    size: str
) -> None:
    """
    Save metadata about the generation to a JSON file.

    Args:
        output_folder: Path to the output folder
        original_prompt: The original user prompt
        enhanced_prompt: The enhanced prompt (if enhancement was used)
        num_variations: Number of variations generated
        enhancement_enabled: Whether enhancement was enabled
        model_name: Name of the model used
        total_cost: Total estimated cost
        width: Image width
        height: Image height
        aspect_ratio: Aspect ratio preset used
        size: Size preset used
    """
    model_info = MODELS[model_name]

    metadata = {
        "original_prompt": original_prompt,
        "enhanced_prompt": enhanced_prompt,
        "model_name": model_name,
        "model_id": model_info["id"],
        "cost_per_image": model_info["cost"],
        "num_variations": num_variations,
        "total_cost": total_cost,
        "timestamp": datetime.now().isoformat(),
        "enhancement_enabled": enhancement_enabled,
        "dimensions": {
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "size": size
        }
    }

    metadata_path = output_folder / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def save_image(image_url: str, output_folder: Path, image_num: int) -> Path:
    """
    Download and save the generated image to the specified folder.

    Args:
        image_url: URL of the generated image
        output_folder: Path to the output folder
        image_num: Image number for filename (e.g., 1, 2, 3)

    Returns:
        Path to the saved file
    """
    filename = f"image_{image_num:02d}.png"
    filepath = output_folder / filename

    # Download and save
    urllib.request.urlretrieve(image_url, filepath)

    return filepath


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate ad creative images using AI"
    )
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",
        help="Text prompt describing the image to generate"
    )
    parser.add_argument(
        "--enhance",
        action="store_true",
        help="Enhance prompt with professional photography terms"
    )
    parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="Use prompt as-is without enhancement"
    )
    parser.add_argument(
        "--variations",
        type=int,
        default=3,
        help="Number of image variations to generate (default: 3, max: 10)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="flux-schnell",
        choices=list(MODELS.keys()),
        help="Model to use for generation (default: flux-schnell)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    parser.add_argument(
        "--aspect-ratio",
        type=str,
        default="1:1",
        choices=list(ASPECT_RATIOS.keys()),
        help="Aspect ratio preset (default: 1:1)"
    )
    parser.add_argument(
        "--size",
        type=str,
        default="large",
        choices=list(SIZES.keys()),
        help="Size preset (default: large)"
    )
    parser.add_argument(
        "--list-sizes",
        action="store_true",
        help="List available sizes and aspect ratios"
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for the image generator."""
    args = parse_arguments()

    # Handle --list-models flag
    if args.list_models:
        list_models()

    # Handle --list-sizes flag
    if args.list_sizes:
        list_sizes()

    # Require prompt if not listing models
    if not args.prompt:
        print("Error: prompt is required")
        print("Use --list-models to see available models")
        sys.exit(1)

    # Validate enhancement flags
    if args.enhance and args.no_enhance:
        print("Error: Cannot use both --enhance and --no-enhance")
        sys.exit(1)

    # Validate variations count
    if args.variations < 1 or args.variations > 10:
        print("Error: --variations must be between 1 and 10")
        sys.exit(1)

    # Load API token
    load_api_token()

    # Get model info
    model_info = MODELS[args.model]
    cost_per_image = model_info["cost"]
    total_cost = args.variations * cost_per_image

    # Calculate dimensions
    width, height = get_dimensions(args.aspect_ratio, args.size)
    dimension_params = get_model_dimension_params(
        args.model, width, height, args.aspect_ratio, args.size
    )

    # Show cost confirmation if cost > $0.01
    if total_cost > 0.01:
        print(f"\nModel: {args.model}")
        print(f"Dimensions: {width}x{height} ({args.aspect_ratio}, {args.size})")
        print(f"Cost per image: ${cost_per_image:.3f}")
        print(f"Total estimated cost: ${total_cost:.3f} ({args.variations} images)")
        response = input("\nProceed? (y/n): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Cancelled.")
            sys.exit(0)

    # Prepare prompts
    original_prompt = args.prompt
    enhanced_prompt = None
    enhancement_enabled = False

    if args.enhance:
        enhanced_prompt = enhance_prompt(original_prompt)
        prompt_to_use = enhanced_prompt
        enhancement_enabled = True
        print(f"\nEnhanced prompt: {prompt_to_use}")
    else:
        prompt_to_use = original_prompt
        print(f"\nUsing prompt: {prompt_to_use}")

    # Create output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_slug = create_prompt_slug(original_prompt)
    folder_name = f"{timestamp}_{prompt_slug}"
    output_folder = Path("output") / folder_name
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating {args.variations} {width}x{height} image(s) with {args.model}...")
    print(f"Output folder: {output_folder}\n")

    try:
        # Generate and save multiple images
        for i in range(1, args.variations + 1):
            # Generate image
            image_url = generate_image(
                prompt_to_use,
                args.model,
                dimension_params,
                width,
                height,
                current=i,
                total=args.variations
            )

            # Save image
            filepath = save_image(image_url, output_folder, i)
            print(f"✓ Saved: {filepath.name}")

        # Save metadata
        save_metadata(
            output_folder,
            original_prompt,
            enhanced_prompt,
            args.variations,
            enhancement_enabled,
            args.model,
            total_cost,
            width,
            height,
            args.aspect_ratio,
            args.size
        )
        print(f"✓ Saved: metadata.json")

        # Display completion message
        print(f"\n✓ Complete! Generated {args.variations} {width}x{height} image(s)")
        print(f"Total cost: ${total_cost:.3f}")

    except Exception as e:
        print(f"\nFailed to generate images: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
