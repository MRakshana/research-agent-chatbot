# index_loader.py

import research_agent as ra

def load_index():
    """
    Wrapper so other scripts can load the same index.
    """
    return ra.build_index()


