import pytest
import matplotlib
import matplotlib.pyplot as plt
from unittest.mock import patch
import traceback

# Set non-interactive backend by default
matplotlib.use('Agg')

def pytest_configure(config):
    """Register show_plots marker"""
    config.addinivalue_line(
        "markers", "show_plots: mark test to run with plots enabled"
    )

# Debug wrapper for plt.show
def debug_show(*args, **kwargs):
    print("plt.show() called from:")
    traceback.print_stack()
    # Don't actually call the original show function
    return None

# Debug wrapper for plt.draw
def debug_draw(*args, **kwargs):
    print("plt.draw() called from:")
    traceback.print_stack()
    # Don't actually call the original draw function
    return None

# Store and replace the original functions
plt.original_show = plt.show
plt.show = debug_show
plt.original_draw = plt.draw
plt.draw = debug_draw

@pytest.fixture(autouse=True)
def configure_plots(request):
    """Fixture to configure plot behavior based on markers"""
    show_plots = request.node.get_closest_marker("show_plots") is not None
    
    if show_plots:
        # If test is marked with show_plots, restore original functions
        print(f"Enabling plots for test: {request.node.name}")
        # Restore original functions
        plt.show = plt.original_show
        plt.draw = plt.original_draw
        yield
    else:
        # Otherwise, disable all plots
        with patch('matplotlib.pyplot.show', return_value=None):
            with patch('matplotlib.pyplot.draw', return_value=None):
                with patch('matplotlib.backend_bases.FigureManagerBase.show', return_value=None):
                    with patch('matplotlib.figure.Figure.show', return_value=None):
                        yield
    
    # Close all figures at the end
    plt.close('all')
