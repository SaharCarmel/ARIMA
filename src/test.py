"""
main.py
====================================
The core module of my example project
"""


def capitalize_name(your_name: str):
    """Decorator that checks NumPy results and CuPy ones are equal.

    Args:
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values are
             appended to the error message.
         name(str): Argument name whose value is either
             ``numpy`` or ``cupy`` module.
         type_check(bool): If ``True``, consistency of dtype is also checked.
         accept_error(bool, Exception or tuple of Exception): Specify
             acceptable errors. When both NumPy test and CuPy test raises the
             same type of errors, and the type of the errors is specified with
             this argument, the errors are ignored and not raised.
             If it is ``True`` all error types are acceptable.
             If it is ``False`` no error is acceptable.
         sp_name(str or None): Argument name whose value is either
             ``scipy.sparse`` or ``cupyx.scipy.sparse`` module. If ``None``, no
             argument is given for the modules.
         scipy_name(str or None): Argument name whose value is either ``scipy``
             or ``cupyx.scipy`` module. If ``None``, no argument is given for
             the modules.
         strides_check(bool): If ``True``, consistency of strides is also
             checked.
    Decorated test fixture is required to return the same arrays
    in the sense of :func:`numpy_cupy_array_equal`
    (except the type of array module) even if ``xp`` is ``numpy`` or ``cupy``.
    .. seealso:: :func:`cupy.testing.assert_array_equal`
    """

    return your_name.capitalize()


def about_me(your_name):
    """
    Return the most important thing about a person.
    Parameters
    ----------
    your_name
        A string indicating the name of the person.
    """
    return "The wise {} loves Python.".format(your_name)


class ExampleClass:
    """An example docstring for a class definition."""

    def __init__(self, name):
        """
        Blah blah blah.
        Parameters
        ---------
        name
            A string to assign to the `name` instance attribute.
        """
        self.name = name

    def about_self(self):
        """
        Return information about an instance created from ExampleClass.
        """
        return "I am a very smart {} object.".format(self.name)
