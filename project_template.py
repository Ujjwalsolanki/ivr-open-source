import os

def create_ivr_project_structure():
    """
    Creates the directory and file structure for the IVR system project.
    """
    project_root = "ivr-local"

    # Define the structure as a list of paths
    # Directories end with a '/'
    # Files do not
    structure = [
        "public/",
        "public/css/",
        "public/js/",
        "public/index.html",
        "public/css/style.css",
        "public/js/main.js",
        "public/js/audio-recorder.js",
        "server/",
        "server/models/",
        "server/models/faster-whisper/",
        "server/models/coqui-tts/",
        "server/models/llama-tiny/",
        "server/modules/",
        "server/main.py",
        "server/requirements.txt",
        "server/config.py",
        "server/modules/stt_module.py",
        "server/modules/tts_module.py",
        "server/modules/llm_module.py",
        "server/modules/ivr_logic.py",
        "README.md"
    ]

    print(f"Creating project structure under: {project_root}/")

    # Create the root directory if it doesn't exist
    os.makedirs(project_root, exist_ok=True)

    for item in structure:
        path = os.path.join(project_root, item)
        if item.endswith('/'):
            # It's a directory
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")
        else:
            # It's a file
            # Ensure its parent directory exists before creating the file
            parent_dir = os.path.dirname(path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
                print(f"Created parent directory for file: {parent_dir}")
            
            with open(path, 'w') as f:
                # Create an empty file
                pass
            print(f"Created file: {path}")

    print("\nProject structure created successfully!")

if __name__ == "__main__":
    create_ivr_project_structure()