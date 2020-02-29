class Debugger:

    def printDebugInfo(self, debug):
        print(self if debug else "", end='')

    def __str__(self, limit=-1, *args, **kwargs):
        # Title separator chars
        title_char = "%" * 5
        # Header separator chars
        sep_char = "-" * 4
        # Add title
        str_obj = "{} {} {}\n".format(title_char, kwargs['title'].upper(), title_char)
        # Iterate over each field
        for index, *debug, in enumerate(zip(*args)):
            # Stop if limit is reached
            if index == limit:
                break
            # Add per-tweet header
            header = "{} {} N. {} {}".format(sep_char, kwargs['header'], index + 1, sep_char)
            # Add debug fields
            str_obj += "{}\n{}\n".format(header, kwargs['template'].format(*debug[0]))
        # Return debug info
        return "{}\n".format(str_obj)
