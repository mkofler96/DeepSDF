import os

# Set the folder where your HTML files are located
folder_path = './notebook_outputs'  # Replace with your folder path

# Get all HTML files in the folder, sort them by filename, and exclude 'index.html'
html_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.html') and f != 'index.html'])

# Create the index.html content
index_html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Main Navigation</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            display: flex;
            margin: 0;
        }}
        .sidebar {{
            width: 500px;
            background-color: #f4f4f4;
            padding: 15px;
            box-shadow: 2px 0 5px rgba(0,0,0,0.1);
            height: 100vh;
        }}
        .sidebar a {{
            display: block;
            padding: 10px;
            margin-bottom: 5px;
            color: #333;
            text-decoration: none;
            border-radius: 4px;
        }}
        .sidebar a:hover {{
            background-color: #ddd;
        }}
        .content {{
            flex-grow: 1;
            padding: 20px;
        }}
        iframe {{
            width: 100%;
            height: 90vh;
            border: none;
        }}
    </style>
</head>
<body>
    <div class="sidebar">
        <h2>Navigation</h2>
        {links}
    </div>
    <div class="content">
        <iframe id="contentFrame" src=""></iframe>
    </div>
    <script>
        document.querySelectorAll('.sidebar a').forEach(link => {{
            link.addEventListener('click', function(e) {{
                e.preventDefault();
                document.getElementById('contentFrame').src = this.getAttribute('href');
            }});
        }});
    </script>
</body>
</html>
"""

# Generate the links for all HTML files
links = '\n'.join([f'<a href="{f}">{f}</a>' for f in html_files])

# Write the index.html file
with open(os.path.join(folder_path, 'index.html'), 'w') as index_file:
    index_file.write(index_html_content.format(links=links))

print("index.html has been created successfully!")
