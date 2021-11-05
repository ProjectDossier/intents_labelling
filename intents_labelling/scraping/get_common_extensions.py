import pandas as pd

url = "https://www.file-extensions.org/extensions/common-file-extension-list"

extensions_list = []
for table in pd.read_html(url):
    try:
        extensions = table[0].tolist()
        extensions_list.extend(extensions)
    except Exception as e:
        print(e)

outfile = "../../data/helpers/common_extensions.txt"
with open(outfile, "w") as fp:
    for line in extensions_list:
        fp.write(f"{str(line).strip()}\n")
