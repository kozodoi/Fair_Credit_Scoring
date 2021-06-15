class DelayedImportError(object):
    """Dummy class to delay raising ImportError until class/method is used.

    Examples:
        >>> try:
        >>>     import foo
        >>> except ImportError:
        >>>     foo = DelayedImportError('foo not installed')
        ...
        >>> try:
        >>>     from foo import bar
        >>> except ImportError:
        >>>     bar = DelayedImportError('foo not installed')
        ...
        >>> print(foo.bar())
        ImportError("foo not installed")
        >>> print(bar())
        ImportError("foo not installed")
    """
    def __init__(self, msg):
        self.msg = msg
    def __call__(self, *args, **kwargs):
        raise ImportError(self.msg)
    def __getattr__(self, attr):
        raise ImportError(self.msg)
