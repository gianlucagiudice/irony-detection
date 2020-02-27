class Debugger:

    def __str__(self, limit=-1, *args, **kwargs):
        sep_char = "-"*5
        str_obj = "%%%% {} %%%%\n".format(kwargs['title'].upper())
        a = list(args)
        for index, *debug in enumerate(zip(*args)):
            p = list(*debug)
            if index == limit:
                break
            header = "{} {} N. {} {}".format(sep_char, kwargs['header'], index, sep_char)
            x = kwargs['template'].format(*p)
            str_obj += "{}\n{}\n".format(header, x)
        return "{}\n".format(str_obj)
