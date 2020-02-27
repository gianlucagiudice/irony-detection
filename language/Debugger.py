class Debugger:

    def __str__(self, limit=-1, *args, **kwargs):
        sep_char = "-" * 4
        title_char = "%" * 5
        str_obj = "{} {} {}\n".format(title_char, kwargs['title'].upper(), title_char)
        v = enumerate(zip(*args))
        for index, *debug, in enumerate(zip(*args)):
            if index == limit:
                break
            header = "{} {} N. {} {}".format(sep_char, kwargs['header'], index + 1, sep_char)
            str_obj += "{}\n{}\n".format(header, kwargs['template'].format(*debug[0]))
        return "{}\n".format(str_obj)
