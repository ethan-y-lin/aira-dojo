import ast
from typing import List, Optional, Dict, Any

def extract_metrics(output: List[str]) -> Optional[Dict[str, Any]]:
    """
    Extracts the final metrics dictionary from a list of strings.
    Looks for the last occurrence of 'METRICS:' and parses the dict following it.
    """
    # Join all strings efficiently
    combined_output = ''.join(output)
    
    # Find last 'METRICS:' occurrence
    idx = combined_output.rfind("METRICS:")
    if idx == -1:
        return None

    # Extract substring after 'METRICS:'
    metrics_str = combined_output[idx + len("METRICS:"):].strip()

    # Try to isolate the dictionary part even if extra text follows
    if '\n' in metrics_str:
        metrics_str = metrics_str.split('\n', 1)[0]

    try:
        return ast.literal_eval(metrics_str)
    except Exception as e:
        print(f"[extract_metrics] Warning: Could not parse metrics — {e}")
        return None