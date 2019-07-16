# This package contains auxiliar functions, useful for debugging and such

# Emojis. Very important stuff

def emoji(key):
    stored = {
    "viva"   : b'\xF0\x9F\x8E\x89'.decode('utf-8'),
    "eyes"   : b'\xF0\x9F\x91\x80'.decode('utf-8'),
    "cycle"  : b'\xF0\x9F\x94\x83'.decode('utf-8'),
    "crying" : b'\xF0\x9F\x98\xAD'.decode('utf-8'),
    "pleft"  : b'\xF0\x9F\x91\x88'.decode('utf-8')
    }
    return stored[key]

# Clean up numerical zeros

def chop(number):
    if abs(number) < 1e-12:
        return 0
    else:
        return number

# Print a pretty matrix

def pretty(inp):
    Mat = inp.tolist()
    out = ''
    for row in Mat:
        for x in row:
            out += ' {:^ 10.7f}'.format(chop(x))
        out += '\n'
    return out
