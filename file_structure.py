import os

def print_python_tree(start_dir, prefix=""):
    # Filter to look at items in the current directory
    items = sorted(os.listdir(start_dir))
    
    # Filter out hidden folders (like .git, .vscode) and virtual environments
    ignored = {'.git', '.vscode', '__pycache__', 'venv', '.venv', 'env'}
    items = [i for i in items if i not in ignored]

    # Keep only folders or .py files
    valid_items = []
    for item in items:
        path = os.path.join(start_dir, item)
        if os.path.isdir(path):
            # Check if directory contains python files at any nested depth
            has_py = any(f.endswith('.py') for _, _, files in os.walk(path) for f in files)
            if has_py:
                valid_items.append(item)
        elif item.endswith('.py'):
            valid_items.append(item)

    # Print the tree layout structure
    for index, item in enumerate(valid_items):
        path = os.path.join(start_dir, item)
        is_last = (index == len(valid_items) - 1)
        connector = "└── " if is_last else "├── "
        
        print(f"{prefix}{connector}{item}")
        
        if os.path.isdir(path):
            next_prefix = prefix + ("    " if is_last else "│   ")
            print_python_tree(path, next_prefix)

if __name__ == "__main__":
    print(".")
    print_python_tree(os.getcwd())
