"""HTML page generator with synthetic frames.

Generates HTML pages with randomly positioned frames and saves metadata.
"""

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import List, Dict


LOREM_IPSUM = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
    "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. "
    "Duis aute irure dolor in reprehenderit in voluptate velit esse. "
    "Excepteur sint occaecat cupidatat non proident, sunt in culpa."
).split()


def generate_lorem_text(min_words: int = 10, max_words: int = 50) -> str:
    """Generate random Lorem Ipsum text."""
    num_words = random.randint(min_words, max_words)
    words = random.sample(LOREM_IPSUM * ((num_words // len(LOREM_IPSUM)) + 1), num_words)
    return " ".join(words)


def generate_random_background() -> Dict:
    """Generate a random background style (solid color or gradient).
    
    Returns:
        Dictionary with background CSS property and type
    """
    background_types = ['solid', 'linear_gradient']
    bg_type = random.choice(background_types)
    
    if bg_type == 'solid':
        # Solid color
        r = random.randint(50, 255)
        g = random.randint(50, 255)
        b = random.randint(50, 255)
        bg_color = f"rgb({r}, {g}, {b})"
        return {
            'type': 'solid',
            'css': bg_color,
        }
    
    else:  # linear_gradient
        # Linear gradient
        angle = random.randint(0, 360)
        # Generate 2-4 colors for gradient
        num_colors = random.randint(2, 4)
        colors = []
        for _ in range(num_colors):
            r = random.randint(50, 255)
            g = random.randint(50, 255)
            b = random.randint(50, 255)
            colors.append(f"rgb({r}, {g}, {b})")
        
        stops = []
        for i, color in enumerate(colors):
            stop_percent = int((i / (len(colors) - 1)) * 100) if len(colors) > 1 else 0
            stops.append(f"{color} {stop_percent}%")
        
        gradient = f"linear-gradient({angle}deg, {', '.join(stops)})"
        return {
            'type': 'linear_gradient',
            'css': gradient,
        }


def generate_frame(page_width: int, page_height: int, frame_id: int) -> Dict:
    """Generate a random frame configuration.
    
    Args:
        page_width: Page width in pixels
        page_height: Page height in pixels
        frame_id: Unique frame ID
        
    Returns:
        Dictionary with frame properties
    """
    # Random size
    min_size = 100
    max_size = min(page_width, page_height) // 2
    w = random.randint(min_size, max_size)
    h = random.randint(min_size, max_size)
    
    # Random position
    x = random.randint(0, max(0, page_width - w))
    y = random.randint(0, max(0, page_height - h))
    
    # Random z-index (will be adjusted in HTML generation)
    z_index = random.randint(1, 100)
    
    # Random border radius (0 to min(w, h)/4)
    max_radius = min(w, h) // 4
    border_radius = random.randint(0, max_radius)
    
    # Random background color
    bg_r = random.randint(200, 255)
    bg_g = random.randint(200, 255)
    bg_b = random.randint(200, 255)
    bg_color = f"rgb({bg_r}, {bg_g}, {bg_b})"
    
    # Random shadow
    shadow_x = random.randint(-5, 5)
    shadow_y = random.randint(-5, 5)
    shadow_blur = random.randint(5, 20)
    shadow_color = f"rgba({random.randint(0, 100)}, {random.randint(0, 100)}, {random.randint(0, 100)}, 0.5)"
    box_shadow = f"{shadow_x}px {shadow_y}px {shadow_blur}px {shadow_color}"
    
    return {
        'id': frame_id,
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'z_index': z_index,
        'border_radius': border_radius,
        'bg_color': bg_color,
        'box_shadow': box_shadow,
    }


def generate_header_footer(page_width: int, page_height: int, min_header_height: int = 60, max_header_height: int = None, 
                           min_footer_height: int = 60, max_footer_height: int = None) -> tuple[Dict, Dict]:
    """Generate header and footer configurations with random heights.
    
    Args:
        page_width: Page width (used for metadata, not CSS)
        page_height: Page height (used to limit max height)
        min_header_height: Minimum header height
        max_header_height: Maximum header height (defaults to page_height // 3)
        min_footer_height: Minimum footer height
        max_footer_height: Maximum footer height (defaults to page_height // 3)
        
    Returns:
        Tuple of (header_dict, footer_dict)
    """
    # Set default max heights if not provided
    if max_header_height is None:
        max_header_height = min(500, 300)
    if max_footer_height is None:
        max_footer_height = min(500, 300)
    
    # Generate random heights
    header_height = random.randint(min_header_height, max_header_height)
    footer_height = random.randint(min_footer_height, max_footer_height)
    
    header = {
        'id': 'header',
        'x': 0,
        'y': 0,
        'w': page_width,  # Used for metadata only
        'h': header_height,
        'z_index': 1,
        'border_radius': 0,
        'position': 'relative',
        'bg_color': 'rgba(30, 30, 30, 0.95)',
        'box_shadow': '0 2px 10px rgba(0, 0, 0, 0.3)',
        'is_header': True,
    }
    
    footer = {
        'id': 'footer',
        'x': 0,
        'y': 0,  # Will be set relative to page height
        'w': page_width,  # Used for metadata only
        'h': footer_height,
        'z_index': 1,
        'border_radius': 0,
        'position': 'relative',
        'bg_color': 'rgba(30, 30, 30, 0.95)',
        'box_shadow': '0 -2px 10px rgba(0, 0, 0, 0.3)',
        'is_footer': True,
    }
    
    return header, footer


def generate_sliders(page_height: int) -> List[Dict]:
    """Generate sliders (sections) for the page, each with a random background.
    
    Args:
        page_height: Total page height
        
    Returns:
        List of slider dictionaries with backgrounds
    """
    # Number of sliders = page_height / 1080 (rounded up)
    num_sliders = max(1, math.ceil(page_height / 1080))
    
    sliders = []
    slider_height = page_height // num_sliders
    
    for i in range(num_sliders):
        slider_bg = generate_random_background()
        # Calculate height for this slider
        slider_top = i * slider_height
        if i == num_sliders - 1:
            # Last slider: use remaining height to cover entire page
            slider_height_actual = page_height - slider_top
        else:
            slider_height_actual = slider_height
        
        sliders.append({
            'id': i + 1,
            'height': slider_height_actual,
            'top': slider_top,
            'background': slider_bg,
        })
    
    return sliders


def generate_html_page(page_id: int, frames: List[Dict], page_width: int, page_height: int, 
                       header: Dict = None, footer: Dict = None, sliders: List[Dict] = None) -> str:
    """Generate HTML content for a page.
    
    Args:
        page_id: Page ID
        frames: List of frame dictionaries
        page_width: Page width
        page_height: Page height
        header: Header configuration dict (optional)
        footer: Footer configuration dict (optional)
        sliders: List of slider dictionaries with backgrounds (optional)
        
    Returns:
        HTML string
    """
    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "    <meta charset='UTF-8'>",
        "    <meta name='viewport' content='width=device-width, initial-scale=1.0, maximum-scale=5.0, user-scalable=yes'>",
        f"    <title>Page {page_id}</title>",
        "    <style>",
        "        * { margin: 0; padding: 0; box-sizing: border-box; }",
        "        html {",
        "            height: 100%;",
        "            width: 100%;",
        "            margin: 0;",
        "            padding: 0;",
        "            overflow-x: hidden;",
        "        }",
        "        body {",
        f"            width: {page_width}px;",
        f"            max-width: 100%;",
        "            margin: 0 auto;",
        f"            min-height: {page_height}px;",
        "            background: transparent;",
        "            font-family: Arial, sans-serif;",
        "            position: relative;",
        "            overflow-x: hidden;",
        "            touch-action: pan-y pinch-zoom;",
        "        }",
        "        .slider {",
        f"            width: {page_width}px;",
        "            position: absolute;",
        "            left: 50%;",
        "            transform: translateX(-50%);",
        "            z-index: 0;",
        "        }",
        "        .header, .footer {",
        "            position: fixed;",
        "            left: 50%;",
        f"            transform: translateX(-50%);",
        f"            width: {page_width}px;",
        "            padding: 20px;",
        "            color: white;",
        "            box-sizing: border-box;",
        "            z-index: 1000;",
        "        }",
        "        .header {",
        "            top: 0;",
        "        }",
        "        .footer {",
        "            bottom: 0;",
        "        }",
        "        main {",
        f"            width: {page_width}px;",
        f"            max-width: 100%;",
        "            margin: 0 auto;",
        f"            min-height: {page_height}px;",
        "            position: relative;",
        "        }",
        "        .frame {",
        "            border: 2px solid rgba(255, 255, 255, 0.3);",
        "            padding: 15px;",
        "            overflow: hidden;",
        "            position: absolute;",
        "        }",
        "        .frame img {",
        "            max-width: 100%;",
        "            height: auto;",
        "            display: block;",
        "            margin-bottom: 10px;",
        "        }",
        "        .frame p {",
        "            color: #333;",
        "            line-height: 1.6;",
        "            font-size: 14px;",
        "            word-wrap: break-word;",
        "            overflow-wrap: break-word;",
        "        }",
        "        @keyframes fadeIn {",
        "            from { opacity: 0; }",
        "            to { opacity: 1; }",
        "        }",
        "        .frame {",
        "            animation: fadeIn 0.5s ease-in;",
        "        }",
        "        /* Адаптивность для планшетов (768px - 1024px) */",
        "        @media screen and (max-width: 1024px) and (min-width: 769px) {",
        "            body {",
        "                width: 100%;",
        "                max-width: 100%;",
        f"                min-height: calc({page_height}px * (100vw / {page_width}));",
        "            }",
        f"            main {{",
        f"                width: {page_width}px;",
        f"                transform: scale(calc(100vw / {page_width}));",
        f"                height: calc({page_height}px * (100vw / {page_width}));",
        "                transform-origin: top center;",
        "            }",
        f"            .slider {{",
        f"                width: {page_width}px;",
        f"                transform: scale(calc(100vw / {page_width}));",
        "                transform-origin: top center;",
        "            }",
        "            .header, .footer {",
        "                width: 100%;",
        "                max-width: 100%;",
        "            }",
        "        }",
        "        /* Адаптивность для маленьких экранов (до 768px) */",
        "        @media screen and (max-width: 768px) {",
        "            body {",
        "                width: 100%;",
        "                max-width: 100%;",
        f"                min-height: calc({page_height}px * (100vw / {page_width}));",
        "            }",
        f"            main {{",
        f"                width: {page_width}px;",
        f"                transform: scale(calc(100vw / {page_width}));",
        f"                height: calc({page_height}px * (100vw / {page_width}));",
        "                transform-origin: top center;",
        "            }",
        f"            .slider {{",
        f"                width: {page_width}px;",
        f"                transform: scale(calc(100vw / {page_width}));",
        "                transform-origin: top center;",
        "            }",
        "            .header, .footer {",
        "                width: 100%;",
        "                max-width: 100%;",
        "                padding: 15px;",
        "            }",
        "            .header h1 {",
        "                font-size: 18px !important;",
        "            }",
        "            .frame {",
        "                padding: 12px;",
        "            }",
        "            .frame p {",
        "                font-size: 13px;",
        "            }",
        "        }",
        "        /* Адаптивность для очень маленьких экранов (до 480px) */",
        "        @media screen and (max-width: 480px) {",
        "            .header, .footer {",
        "                padding: 10px;",
        "            }",
        "            .header h1 {",
        "                font-size: 16px !important;",
        "            }",
        "            .frame {",
        "                padding: 10px;",
        "            }",
        "            .frame p {",
        "                font-size: 12px;",
        "            }",
        "        }",
        "        /* Адаптивность для средних экранов (1025px - 1440px) */",
        f"        @media screen and (min-width: 1025px) and (max-width: {min(page_width - 1, 1440)}px) {{",
        "            body {",
        "                width: 100%;",
        "                max-width: 100%;",
        f"                min-height: calc({page_height}px * (100vw / {page_width}));",
        "            }",
        f"            main {{",
        f"                width: {page_width}px;",
        f"                transform: scale(calc(100vw / {page_width}));",
        f"                height: calc({page_height}px * (100vw / {page_width}));",
        "                transform-origin: top center;",
        "            }",
        f"            .slider {{",
        f"                width: {page_width}px;",
        f"                transform: scale(calc(100vw / {page_width}));",
        "                transform-origin: top center;",
        "            }",
        "            .header, .footer {",
        "                width: 100%;",
        "                max-width: 100%;",
        "            }",
        "        }",
        "        /* Адаптивность для широких экранов (1441px - 1919px) */",
        "        @media screen and (min-width: 1441px) and (max-width: 1919px) {",
        "            body {",
        f"                width: {page_width}px;",
        "                max-width: 100%;",
        "            }",
        "            main {",
        f"                width: {page_width}px;",
        "                max-width: 100%;",
        "            }",
        "            .header, .footer {",
        f"                width: {page_width}px;",
        "                max-width: 100%;",
        "            }",
        "        }",
        "        /* Адаптивность для очень широких экранов (от 1920px) */",
        "        @media screen and (min-width: 1920px) {",
        "            body {",
        f"                width: {page_width}px;",
        f"                max-width: {page_width}px;",
        "            }",
        "            main {",
        f"                width: {page_width}px;",
        f"                max-width: {page_width}px;",
        "            }",
        "            .header, .footer {",
        f"                width: {page_width}px;",
        f"                max-width: {page_width}px;",
        "            }",
        "        }",
        "    </style>",
        "</head>",
        "<body>",
    ]
    
    # Add sliders with different backgrounds
    if sliders:
        for slider in sliders:
            slider_style = [
                f"height: {slider['height']}px;",
                f"top: {slider['top']}px;",
                f"background: {slider['background']['css']};",
            ]
            html_parts.append(f"    <div class='slider' style='{'; '.join(slider_style)}'></div>")
    
    # Add header if provided
    if header:
        header_style = [
            f"background-color: {header['bg_color']};",
            f"box-shadow: {header['box_shadow']};",
            f"height: {header['h']}px;",
        ]
        html_parts.append(f"    <header class='header' style='{'; '.join(header_style)}'>")
        html_parts.append(f"        <h1 style='color: white; font-size: 24px;'>Header - Page {page_id}</h1>")
        html_parts.append("    </header>")
    
    # Add main content area
    html_parts.append("    <main>")
    
    # Add frames positioned relative to main (absolute positioning)
    for frame in frames:
        style_parts = [
            "position: absolute;",
            f"left: {frame['x']}px;",
            f"top: {frame['y']}px;",
            f"width: {frame['w']}px;",
            f"height: {frame['h']}px;",
            # z-index: фреймы выше sliders (0), но ниже header/footer (1000)
            f"z-index: {min(frame['z_index'] + 100, 999)};",
            f"background-color: {frame['bg_color']};",
            f"border-radius: {frame['border_radius']}px;",
            f"box-shadow: {frame['box_shadow']};",
        ]
        
        img_w = min(frame['w'] - 30, 200)
        img_h = min(frame['h'] // 3, 150)
        img_seed = page_id * 1000 + frame['id']
        img_url = f"https://picsum.photos/{img_w}/{img_h}?random={img_seed}"
        
        text = generate_lorem_text(5, 30)
        
        html_parts.append(f"        <div class='frame' style='{'; '.join(style_parts)}'>")
        html_parts.append(f"            <img src='{img_url}' alt='Random image' />")
        html_parts.append(f"            <p>{text}</p>")
        html_parts.append("        </div>")
    
    html_parts.append("    </main>")
    
    # Add footer if provided
    if footer:
        footer_style = [
            f"background-color: {footer['bg_color']};",
            f"box-shadow: {footer['box_shadow']};",
            f"height: {footer['h']}px;",
        ]
        html_parts.append(f"    <footer class='footer' style='{'; '.join(footer_style)}'>")
        html_parts.append(f"        <p style='color: white; text-align: center;'>Footer - Page {page_id}</p>")
        html_parts.append("    </footer>")
    
    html_parts.extend([
        "</body>",
        "</html>",
    ])
    
    return "\n".join(html_parts)


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic HTML pages with frames')
    parser.add_argument('--n', type=int, default=3, help='Number of pages to generate')
    parser.add_argument('--min-frames', type=int, default=10, help='Minimum frames per page')
    parser.add_argument('--max-frames', type=int, default=20, help='Maximum frames per page')
    parser.add_argument('--page-width', type=int, default=1920, help='Page width in pixels')
    parser.add_argument('--page-height', type=int, default=1080, help='Page height in pixels')
    parser.add_argument('--output-dir', type=str, default='data', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directories
    pages_dir = Path(args.output_dir) / 'pages'
    meta_dir = Path(args.output_dir) / 'meta'
    pages_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {args.n} pages...")
    
    for page_id in range(1, args.n + 1):
        # Generate header and footer FIRST to know their heights
        header, footer = generate_header_footer(args.page_width, args.page_height)
        header_height = header['h']
        footer_height = footer['h']
        
        # Calculate available height for frames (excluding header and footer)
        available_height = args.page_height - header_height - footer_height
        
        # Generate random number of frames
        num_frames = random.randint(args.min_frames, args.max_frames)
        
        # Generate frames with consideration for header/footer
        frames = []
        for i in range(num_frames):
            # Generate frame in available area
            frame = generate_frame(args.page_width, available_height, i + 1)
            # Shift y-coordinate by header height so frames don't overlap header
            frame['y'] += header_height
            frames.append(frame)
        
        # Calculate actual page width based on frames (rightmost edge of all frames)
        actual_page_width = args.page_width
        if frames:
            max_right_edge = max(frame['x'] + frame['w'] for frame in frames)
            # Use the maximum between configured width and actual content width
            actual_page_width = max(args.page_width, max_right_edge)
        
        # Ensure frames don't exceed actual page width
        for frame in frames:
            if frame['x'] + frame['w'] > actual_page_width:
                frame['x'] = max(0, actual_page_width - frame['w'])
        
        # Update header and footer width (keep same heights)
        header['w'] = actual_page_width
        footer['w'] = actual_page_width
        
        # Generate sliders with random backgrounds
        # Number of sliders = page_height / 1080 (rounded up)
        sliders = generate_sliders(args.page_height)
        
        # Mark frames that are in header/footer areas (for metadata purposes)
        for frame in frames:
            # Check if frame is in header area
            if frame['y'] + frame['h'] <= header_height:
                frame['in_header'] = True
            else:
                frame['in_header'] = False
            
            # Check if frame is in footer area
            if frame['y'] >= args.page_height - footer_height:
                frame['in_footer'] = True
            else:
                frame['in_footer'] = False
        
        # Generate HTML with actual page width
        html_content = generate_html_page(page_id, frames, actual_page_width, args.page_height, header, footer, sliders)
        
        # Save HTML
        html_path = pages_dir / f"page_{page_id}.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Save metadata (header and footer are saved but marked as non-mask elements)
        metadata = {
            'page_id': page_id,
            'page_width': args.page_width,
            'actual_page_width': actual_page_width,
            'page_height': args.page_height,
            'header': header,
            'footer': footer,
            'sliders': sliders,
            'frames': frames,
        }
        
        meta_path = meta_dir / f"page_{page_id}.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        if page_id % 10 == 0:
            print(f"Generated {page_id}/{args.n} pages")
    
    print(f"Done! Generated {args.n} pages in {pages_dir}")
    print(f"Metadata saved in {meta_dir}")


if __name__ == '__main__':
    main()

