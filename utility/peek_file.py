import pickle
import pandas as pd
import numpy as np
from pathlib import Path

def explore_pickle_file(filepath):
    """
    Comprehensive overview of a pickle file contents
    """
    print(f"Exploring pickle file: {filepath}")
    print("=" * 50)
    
    try:
        # Load the pickle file
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Basic information
        print(f"Data type: {type(data)}")
        print(f"Data class: {data.__class__.__name__}")
        
        # Size information
        file_size = Path(filepath).stat().st_size
        print(f"File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        
        # Detailed exploration based on data type
        if isinstance(data, dict):
            explore_dict(data)
        elif isinstance(data, list):
            explore_list(data)
        elif isinstance(data, tuple):
            explore_tuple(data)
        elif isinstance(data, pd.DataFrame):
            explore_dataframe(data)
        elif isinstance(data, np.ndarray):
            explore_numpy_array(data)
        else:
            explore_other_object(data)
            
    except Exception as e:
        print(f"Error loading pickle file: {e}")

def explore_dict(data):
    """Explore dictionary contents"""
    print(f"\n📁 DICTIONARY OVERVIEW")
    print(f"Number of keys: {len(data)}")
    print(f"Keys: {list(data.keys())}")
    
    print(f"\n🔍 KEY-VALUE DETAILS:")
    for i, (key, value) in enumerate(data.items()):
        print(f"  [{i+1}] '{key}': {type(value)} - {get_size_info(value)}")
        if hasattr(value, 'shape'):
            print(f"      Shape: {value.shape}")
        if isinstance(value, (list, tuple)) and len(value) > 0:
            print(f"      First item type: {type(value[0])}")
        if isinstance(value, str) and len(value) < 100:
            print(f"      Content: {repr(value)}")

def explore_list(data):
    """Explore list contents"""
    print(f"\n📋 LIST OVERVIEW")
    print(f"Length: {len(data)}")
    if len(data) > 0:
        print(f"First item type: {type(data[0])}")
        print(f"Last item type: {type(data[-1])}")
        
        # Check if all items are same type
        types = set(type(item) for item in data)
        if len(types) == 1:
            print(f"All items are: {list(types)[0]}")
        else:
            print(f"Mixed types: {types}")
            
        # Show first few items
        print(f"\n🔍 FIRST 5 ITEMS:")
        for i, item in enumerate(data[:5]):
            print(f"  [{i}] {type(item)}: {get_preview(item)}")

def explore_tuple(data):
    """Explore tuple contents"""
    print(f"\n📦 TUPLE OVERVIEW")
    print(f"Length: {len(data)}")
    print(f"\n🔍 ITEMS:")
    for i, item in enumerate(data):
        print(f"  [{i}] {type(item)}: {get_preview(item)}")

def explore_dataframe(data):
    """Explore pandas DataFrame"""
    print(f"\n📊 DATAFRAME OVERVIEW")
    print(f"Shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"Data types:\n{data.dtypes}")
    print(f"\n🔍 FIRST 5 ROWS:")
    print(data.head())
    print(f"\n📈 BASIC STATS:")
    print(data.describe())

def explore_numpy_array(data):
    """Explore numpy array"""
    print(f"\n🔢 NUMPY ARRAY OVERVIEW")
    print(f"Shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Size: {data.size:,} elements")
    print(f"Memory usage: {data.nbytes:,} bytes")
    print(f"\n🔍 ARRAY PREVIEW:")
    if data.size < 50:
        print(data)
    else:
        print("First 10 elements:", data.flat[:10])
        print("Shape details:", data.shape)

def explore_other_object(data):
    """Explore other object types"""
    print(f"\n🔍 OBJECT OVERVIEW")
    print(f"Type: {type(data)}")
    print(f"String representation: {str(data)[:200]}...")
    
    # Try to get attributes
    try:
        attrs = [attr for attr in dir(data) if not attr.startswith('_')]
        if attrs:
            print(f"Available attributes: {attrs[:10]}")  # Show first 10
    except:
        pass

def get_size_info(obj):
    """Get size information for an object"""
    if hasattr(obj, '__len__'):
        return f"length {len(obj)}"
    elif hasattr(obj, 'shape'):
        return f"shape {obj.shape}"
    else:
        return f"size ~{len(str(obj))} chars"

def get_preview(obj):
    """Get a preview of an object"""
    if isinstance(obj, str):
        return repr(obj[:50] + "..." if len(obj) > 50 else obj)
    elif isinstance(obj, (int, float, bool)):
        return str(obj)
    elif hasattr(obj, 'shape'):
        return f"array with shape {obj.shape}"
    elif hasattr(obj, '__len__'):
        return f"{type(obj).__name__} with {len(obj)} items"
    else:
        return str(obj)[:50] + "..."

# Usage example:
if __name__ == "__main__":
    # Replace with your actual file path
    pickle_files = [
        "data_0.pkl",
        "data_1.pkl", 
        "data_2.pkl",
        "log.log"  # Note: .log files are typically text, not pickle
    ]
    
    for filepath in pickle_files:
        if filepath.endswith('.pkl'):
            try:
                explore_pickle_file(filepath)
                print("\n" + "="*80 + "\n")
            except FileNotFoundError:
                print(f"File not found: {filepath}")
            except Exception as e:
                print(f"Error with {filepath}: {e}")

# Quick exploration function for single file
def quick_peek(filepath):
    """Quick peek at pickle file contents"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    print(f"File: {filepath}")
    print(f"Type: {type(data)}")
    if hasattr(data, 'shape'):
        print(f"Shape: {data.shape}")
    elif hasattr(data, '__len__'):
        print(f"Length: {len(data)}")
    
    return data

# Example usage:
# data = quick_peek("data_0.pkl")
# explore_pickle_file("data_0.pkl")
