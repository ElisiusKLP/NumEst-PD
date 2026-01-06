from pathlib import Path

def get_paths(dir: str, glob_str: str = '*'):
    """
    Return a list of all file paths of csv data
    """

    base = Path.cwd()
    data_dir = base / dir
    data_files = list(data_dir.glob(glob_str))
    print(f"data_files {data_files}")
    
    if len(data_files) == 1:
        print("unpacked list")
        return data_files[0]
    if len(data_files) == 0:
        raise ValueError("No paths returned")
    
    return data_files