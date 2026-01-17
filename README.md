# OneMetrik Image Generator (omgen)

A CLI tool for generating ad creative variations using the Replicate API with the Flux Schnell model.

## What This Tool Does

This tool allows you to generate multiple variations of ad creatives from text prompts using AI image generation. It's designed for quickly creating test ad variations for marketing campaigns.

## Installation

1. Clone this repository or download the files

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your Replicate API token:
   - Copy `.env.example` to `.env`
   - Get your API token from [replicate.com](https://replicate.com)
   - Add your token to the `.env` file:
     ```
     REPLICATE_API_TOKEN=your_actual_token_here
     ```

## Usage

Make sure your virtual environment is activated:
```bash
source venv/bin/activate
```

Then run the generator:
```bash
python src/generate.py "your prompt here" --variations 5
```

Generated images will be saved to the `output/` directory.

## Requirements

- Python 3.11 or higher
- Replicate API account and token
- Internet connection for API calls

## Project Structure

- `src/` - Source code
- `output/` - Generated images (not tracked in git)
- `tests/` - Test files
- `.env` - Your API credentials (not tracked in git)

## Security Reminder

**Never commit `.env` or any file containing API tokens/secrets.**

- The `.env` file is gitignored and should stay that way
- Always use `.env.example` with placeholder values only
- If you accidentally commit a secret, revoke it immediately and clean git history

## Note

This is a learning project for API integration. Please be mindful of API costs when generating images.
