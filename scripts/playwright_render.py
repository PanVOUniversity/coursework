"""Playwright renderer for taking full-page screenshots.

Renders HTML pages with Playwright, disables animations, scrolls to load content,
and takes full-page screenshots.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from tqdm import tqdm


async def render_page(page, html_path: Path, output_path: Path, timeout: int = 30000):
    """Render a single HTML page and take a screenshot.
    
    Args:
        page: Playwright page object
        html_path: Path to HTML file
        output_path: Path to save screenshot
        timeout: Timeout in milliseconds
    """
    try:
        # Load HTML file
        file_url = f"file://{html_path.absolute()}"
        await page.goto(file_url, wait_until='domcontentloaded', timeout=timeout)
        
        # Disable animations
        await page.add_style_tag(content='*{animation:none!important;transition:none!important;}')
        
        # Wait a bit for images to load
        await page.wait_for_timeout(1000)
        
        # Scroll to load all content
        viewport_height = page.viewport_size['height']
        last_height = 0
        scroll_attempts = 0
        max_scroll_attempts = 50
        
        while scroll_attempts < max_scroll_attempts:
            # Get current scroll height
            current_height = await page.evaluate('document.body.scrollHeight')
            
            if current_height == last_height:
                # Height stabilized, check if we're at bottom
                scroll_y = await page.evaluate('window.scrollY')
                max_scroll = current_height - viewport_height
                if scroll_y >= max_scroll - 10:
                    break
            
            # Scroll down
            await page.evaluate(f'window.scrollBy(0, {viewport_height})')
            await page.wait_for_timeout(500)  # Wait for content to load
            await page.wait_for_load_state('networkidle', timeout=5000)
            
            last_height = current_height
            scroll_attempts += 1
        
        # Scroll back to top
        await page.evaluate('window.scrollTo(0, 0)')
        await page.wait_for_timeout(500)
        
        # Take full page screenshot
        await page.screenshot(path=str(output_path), full_page=True, timeout=timeout)
        
        return True
    except PlaywrightTimeoutError:
        print(f"Timeout rendering {html_path}")
        return False
    except Exception as e:
        print(f"Error rendering {html_path}: {e}")
        return False


def get_page_viewport(html_path: Path, meta_dir: Path, default_width: int = 1920, default_height: int = 1080):
    """Get viewport size from metadata or use defaults.
    
    Args:
        html_path: Path to HTML file
        meta_dir: Directory with metadata JSON files
        default_width: Default viewport width
        default_height: Default viewport height
        
    Returns:
        Tuple of (width, height)
    """
    page_id = html_path.stem.replace('page_', '')
    meta_path = meta_dir / f"page_{page_id}.json"
    
    if meta_path.exists():
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            width = metadata.get('page_width', default_width)
            height = metadata.get('page_height', default_height)
            return (width, height)
        except Exception as e:
            print(f"Warning: Could not read metadata from {meta_path}: {e}")
    
    return (default_width, default_height)


async def render_pages_async(html_files: list, output_dir: Path, meta_dir: Path, 
                             workers: int = 4, default_width: int = 1920, default_height: int = 1080):
    """Render multiple pages asynchronously.
    
    Args:
        html_files: List of HTML file paths
        output_dir: Output directory for screenshots
        meta_dir: Directory with metadata JSON files
        workers: Number of concurrent workers
        default_width: Default viewport width if metadata not found
        default_height: Default viewport height if metadata not found
    """
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=True)
        
        # Create semaphore to limit concurrent workers
        semaphore = asyncio.Semaphore(workers)
        
        async def render_with_semaphore(html_path):
            async with semaphore:
                # Get viewport size from metadata
                viewport_width, viewport_height = get_page_viewport(
                    html_path, meta_dir, default_width, default_height
                )
                
                context = await browser.new_context(
                    viewport={'width': viewport_width, 'height': viewport_height},
                    device_scale_factor=1
                )
                page = await context.new_page()
                
                page_id = html_path.stem.replace('page_', '')
                output_path = output_dir / f"page_{page_id}.png"
                
                success = await render_page(page, html_path, output_path)
                
                await context.close()
                return success
        
        # Render all pages
        tasks = [render_with_semaphore(html_path) for html_path in html_files]
        results = await asyncio.gather(*tasks)
        
        await browser.close()
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Render HTML pages with Playwright')
    parser.add_argument('--input-dir', type=str, default='data/pages', help='Input directory with HTML files')
    parser.add_argument('--output-dir', type=str, default='data/screenshots', help='Output directory for screenshots')
    parser.add_argument('--meta-dir', type=str, default='data/meta', help='Directory with metadata JSON files')
    parser.add_argument('--workers', type=int, default=4, help='Number of concurrent workers')
    parser.add_argument('--viewport-width', type=int, default=1920, help='Default viewport width (used if metadata not found)')
    parser.add_argument('--viewport-height', type=int, default=1080, help='Default viewport height (used if metadata not found)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    meta_dir = Path(args.meta_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all HTML files
    html_files = sorted(input_dir.glob('page_*.html'))
    
    if not html_files:
        print(f"No HTML files found in {input_dir}")
        return
    
    print(f"Found {len(html_files)} HTML files")
    print(f"Rendering with {args.workers} workers...")
    print(f"Viewport sizes will be read from metadata in {meta_dir}")
    print(f"Default viewport: {args.viewport_width}x{args.viewport_height} (if metadata not found)")
    
    # Render pages
    results = asyncio.run(render_pages_async(
        html_files, output_dir, meta_dir, args.workers,
        args.viewport_width, args.viewport_height
    ))
    
    success_count = sum(results)
    print(f"Done! Successfully rendered {success_count}/{len(html_files)} pages")
    print(f"Screenshots saved in {output_dir}")


if __name__ == '__main__':
    main()

