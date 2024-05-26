__all__ = [
    "docstring_decorator"
    "doc_obj_coords",
    "doc_img_coords",
    "doc_cam_struct"
]

# typing the same doc string multiple times geets annoying and error prone...
def docstring_decorator(*args, **kwargs):
    def decorate(obj):
        obj.__doc__ = obj.__doc__.format(*args, **kwargs)
        return obj
    
    return decorate

# and here are some strings that are repetative.
doc_obj_coords = """Real world coordinates. The ndarray is structured like [X, Y]'."""

doc_img_coords = """Image coordinates. The ndarray is structured like [x, y]'."""

doc_cam_struct = """A dictionary structure of camera parameters."""