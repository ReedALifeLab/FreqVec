#puts docs compiled from various SIC group descriptions into a special folder, for use in WV models.
import os
import string

TOPATH = "test_data"
divisions = "A B C D E F G H I J".split()
# major_groups = 
division_ranges = {"A":(100,999),"B":(1000,1499),"C":(1500,1799),"D":(2000,3999),"E":(4000,4999),"F":(5000,5199),"G":(5200,5999),"H":(6000,6799),"I":(7000,8999),"J":(9100,9729)}

def compile_sic_groups(FROMPATH, grouplevel = 'division', runID = ""):
    """
    :FROMPATH: 
        string, the folder where the processed sic descriptions are stored
    :grouplvel:
        "div" -> divisions
        "mg" -> major groups
        tells the program which groups to compile.
    """
    sic_processed = os.listdir(FROMPATH)
    if grouplevel == 'division':
        for division in divisions:
            divtitle = ((4 - len(str(division_ranges[division][0]))) * "0" + str(division_ranges[division][0])) + "_division_" + division + runID
            divtext = open(TOPATH + "/" + divtitle, "w", encoding="utf-8")
            divtext.write(divtitle + ", ")
            for filename in sic_processed:
                if "." in filename:
                    fn = filename[:-4]
                else:
                    fn = filename
                # print(filename)
                if fn == division:
                    file = open(FROMPATH + "/" + filename, "r", encoding="utf-8")
                    filetext = file.read()
                    divtext.write(filetext + " ")
                    file.close()
                elif fn not in string.ascii_uppercase:
                    filename_pad = fn + "0" * (4 - len(fn))
                    if(int(filename_pad) <= division_ranges[division][1] and int(filename_pad) >= division_ranges[division][0]):
                        file = open(FROMPATH + "/" + filename, "r", encoding="utf-8")
                        filetext = file.read()
                        divtext.write(filetext + "\n")
                        file.close()
            divtext.close()

compile_sic_groups("sic_descriptions")