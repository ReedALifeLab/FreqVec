#fills test_data folder with properly named files, given a raw output csv.

TOPATH = "test_data"

def splitfile(FROMPATH):
    skipped_labels = False
    line_number = 1
    with open(FROMPATH, 'r', encoding='utf-8') as infile:
        for text in infile:
            # print("Reading line " + str(line_number))
            line_number += 1
            if skipped_labels:
                ID = ""
                for char in text:
                    if char not in '"[],/':
                        ID += char
                    if char in ",/":
                        break
                # if ID[5:15] == '0000789019' or ID[5:15] == '0000886982' or ID[5:15] == '0000019617':
                outfile = open(TOPATH + "/" + ID, 'w', encoding='utf-8')
                outfile.write(text)
                outfile.close()
            else:
                skipped_labels = True
    print("split input file")