import os
import html
import re

def generate_index_html(root_dir):
    """
    Generates an index.html file in the root_dir that contains links to all files
    in the directory tree, organized by folder structure.
    Includes 'Open All Files' button for each folder and properly sorts filenames numerically.
    
    Args:
        root_dir (str): Path to the visualization folder
    """
    
    output_file = os.path.join(root_dir, "index.html")
    
    html_content = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "    <meta charset='UTF-8'>",
        "    <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "    <title>Visualization Directory Index</title>",
        "    <style>",
        "        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }",
        "        h1 { color: #333; }",
        "        details { margin-bottom: 10px; }",
        "        summary { cursor: pointer; font-weight: bold; color: #444; padding: 8px; }",
        "        summary:hover { background-color: #f5f5f5; }",
        "        ul { list-style-type: none; padding-left: 20px; margin-top: 5px; }",
        "        li { margin: 5px 0; }",
        "        a { color: #0066cc; text-decoration: none; }",
        "        a:hover { text-decoration: underline; }",
        "        .file { }",
        "        .folder-path { font-weight: bold; color: #555; }",
        "        .container { max-width: 1200px; margin: 0 auto; }",
        "        .level-1 { margin-left: 0px; }",
        "        .level-2 { margin-left: 20px; }",
        "        .level-3 { margin-left: 40px; }",
        "        .level-4 { margin-left: 60px; }",
        "        .level-5 { margin-left: 80px; }",
        "        .level-6 { margin-left: 100px; }",
        "        .level-7 { margin-left: 120px; }",
        "        .level-8 { margin-left: 140px; }",
        "        .open-all-btn {",
        "            background-color: #4CAF50;",
        "            color: white;",
        "            padding: 5px 10px;",
        "            border: none;",
        "            border-radius: 4px;",
        "            cursor: pointer;",
        "            margin: 5px 0;",
        "            font-size: 14px;",
        "        }",
        "        .open-all-btn:hover {",
        "            background-color: #45a049;",
        "        }",
        "    </style>",
        "    <script>",
        "        function openAllFiles(folderId) {",
        "            const links = document.querySelectorAll('#' + folderId + ' a');",
        "            links.forEach(link => {",
        "                window.open(link.href, '_blank');",
        "            });",
        "        }",
        "    </script>",
        "</head>",
        "<body>",
        "    <div class='container'>",
        "        <h1>Visualization Directory Index</h1>"
    ]

    def natural_sort_key(s):
        """
        Function to generate a key for sorting strings in natural order.
        For example: ['1', '2', '10'] instead of ['1', '10', '2']
        """
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
    
    # Counter for unique folder IDs
    folder_counter = 0
    
    # Define layer sort function here for use in both functions
    def layer_sort_key(filename):
        layer_match = re.match(r'model\.layers\.(\d+)\.mlp\.html', filename)
        if layer_match:
            return (0, int(layer_match.group(1)))  # Sort these by layer number
        return (1, natural_sort_key(filename))  # Other files come after
    
    def create_folder_section(directory, level, display_name, files, dirs, relative_path):
        """Create HTML section for a folder"""
        nonlocal folder_counter
        
        # Only create a section if there are files or directories to show
        if files or dirs:
            level_class = f"level-{min(level + 1, 8)}"
            
            # Create a details section for this directory
            html_content.append(f"{'    ' * level}<details class='{level_class}' {'open' if level == 0 else ''}>")
            html_content.append(f"{'    ' * level}    <summary class='folder-path'>{html.escape(display_name)}</summary>")
            
            # Add files with "Open All" button if there are files
            if files:
                folder_id = f"folder_{folder_counter}"
                folder_counter += 1
                
                html_content.append(f"{'    ' * level}    <button class='open-all-btn' onclick=\"openAllFiles('{folder_id}')\">Open All Files</button>")
                html_content.append(f"{'    ' * level}    <ul id='{folder_id}'>")
                
                # Sort files
                for file_name in sorted(files, key=layer_sort_key):
                    file_path = os.path.join(relative_path, file_name) if relative_path else file_name
                    html_content.append(f"{'    ' * (level+1)}    <li class='file'><a href='{html.escape(file_path)}'>{html.escape(file_name)}</a></li>")
                
                html_content.append(f"{'    ' * level}    </ul>")
            
            # Process subdirectories
            for dir_name in dirs:
                sub_path = os.path.join(directory, dir_name)
                sub_rel_path = os.path.join(relative_path, dir_name) if relative_path else dir_name
                process_directory(sub_path, level + 1, sub_rel_path)
            
            html_content.append(f"{'    ' * level}</details>")
    
    def process_directory_with_name(directory, level, relative_path, display_name):
        """Process a directory with a custom display name"""
        
        # Get all files and directories
        files = []
        dirs = []
        
        try:
            items = os.listdir(directory)
            items.sort(key=natural_sort_key)
            
            for item in items:
                item_path = os.path.join(directory, item)
                
                if os.path.isdir(item_path):
                    dirs.append(item)
                else:
                    files.append(item)
        except (PermissionError, FileNotFoundError):
            return
        
        # Check if this is an empty intermediate folder (no files, just one subfolder)
        if not files and len(dirs) == 1:
            # Skip this level and process the single subfolder directly
            sub_dir = dirs[0]
            sub_path = os.path.join(directory, sub_dir)
            sub_rel_path = os.path.join(relative_path, sub_dir)
            
            # Continue combining folder names
            combined_path = os.path.join(display_name, sub_dir)
            process_directory_with_name(sub_path, level, sub_rel_path, combined_path)
        else:
            # Create a normal folder section
            create_folder_section(directory, level, display_name, files, dirs, relative_path)
    
    def process_directory(directory, level=0, relative_path=""):
        """Process a directory and return its HTML content"""
        
        # Skip the index.html file and the script itself
        ignore_files = ["index.html", "generate_index.py"]
        
        # Define display name for the current directory - just show basename
        display_name = "Root" if relative_path == "" else os.path.basename(relative_path)
        
        # Get all files and directories
        files = []
        dirs = []
        
        try:
            items = os.listdir(directory)
            # Sort items for consistent display
            items.sort(key=natural_sort_key)
            
            for item in items:
                item_path = os.path.join(directory, item)
                
                # Skip ignored files in root directory
                if item in ignore_files and directory == root_dir:
                    continue
                
                if os.path.isdir(item_path):
                    dirs.append(item)
                else:
                    files.append(item)
        except PermissionError:
            html_content.append(f"{'    ' * level}<p>Permission denied to read directory: {html.escape(display_name)}</p>")
            return
        except FileNotFoundError:
            html_content.append(f"{'    ' * level}<p>Directory not found: {html.escape(display_name)}</p>")
            return
        
        # Check if this is an empty intermediate folder (no files, just one subfolder)
        if not files and len(dirs) == 1:
            # Skip this level and process the single subfolder directly
            sub_dir = dirs[0]
            sub_path = os.path.join(directory, sub_dir)
            sub_rel_path = os.path.join(relative_path, sub_dir) if relative_path else sub_dir
            
            # Use combined name: current folder + subfolder
            combined_path = os.path.join(display_name, sub_dir) if display_name != "Root" else sub_dir
            process_directory_with_name(sub_path, level, sub_rel_path, combined_path)
        else:
            # Create a normal folder section
            create_folder_section(directory, level, display_name, files, dirs, relative_path)
    
    # Start processing from the root directory
    process_directory(root_dir)
    
    # Close HTML
    html_content.extend([
        "    </div>",
        "</body>",
        "</html>"
    ])
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_content))
    
    print(f"Index file generated at: {output_file}")

if __name__ == "__main__":
    # Get the visualization folder path (current directory or specify path)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # We assume this script is placed in the visualization folder
    visualization_dir = script_dir
    
    # Generate the index
    generate_index_html(visualization_dir)