import numpy as np
import pandas

def yearly_with_padding(INPUTPATH, OUTPUTPATH, STARTYEAR, STOPYEAR):
# INPUTPATH = "all_tfidf_1.0_dow_v5_similarities.csv"
# STARTYEAR = 1993
# STOPYEAR = 2002

    M = pandas.read_csv(INPUTPATH, index_col=0)
    # _labels = {label:"0000" + label for label in M.columns if label[0] == "_"}
    # M.rename(_labels, axis=1, inplace=True)
# included_ciks = [i for i in M.columns if i[i.index("_")+1:i.index("_") + 11] in np.load("2001_2018_continuous_filings_for_v4.npy")]
# cik_repr = M[included_ciks].T[included_ciks].T
# print(cik_repr)

    ciks_to_sics = {i[5:15]:i[0:4] for i in M.columns} 

# # _labels = {label:"0000" + label for label in M.columns if label[0] == "_"}
# # M.rename(_labels, axis=0)
# # M.rename(_labels, axis=1)
# ciks_to_years = {}
# for label in M.columns:
#     # year = label[16:20]
#     # cik = label[5:15]

#     # label index _year
#     # label index _000
#     # but easier if i just use tobias' list of companies

#     if int(year) >= STARTYEAR and int(year) < STOPYEAR:
#         if cik in ciks_to_years:
#             ciks_to_years[cik].add(year)
#         else:
#             ciks_to_years[cik] = {year}
# cik_repr = {cik for cik in ciks_to_years if len(ciks_to_years[cik]) == STOPYEAR - STARTYEAR}
# cik_restricted = M[[i for i in M.columns if i[5:15] in cik_repr]].T[[i for i in M.columns if i[5:15] in cik_repr]].T

# printnum = 0

# for year in range(STARTYEAR, STOPYEAR):
#     seen_ciks = {}
#     year_repr = {}
#     for i in M.columns:
#         if "_" + str(year) in i:
#             seen_ciks.add(i[5:15])
#             year_repr[i] = M[i]
#     for cik in ciks_to_sics:
#         if cik not in seen_ciks:
#             newcolname = ciks_to_sics[cik] + "_" + cik + "_missing_for_" + str(year)
#             year_repr
#     m = M[year_repr].T[year_repr].T
#     for cik in ciks_to_sics:
#         if cik not in m.columns:
#             newcolname = ciks_to_sics[cik] + "_" + cik + "_" + str(year) + "_missing"
#             m.append(pandas.DataFrame([np.nan for k in m.columns], columns=m.columns, )
#                 {k:[np.nan] for k in m.columns}, ignore_index=True)
#             m[newcolname] = [np.nan for i in range(len(m.index))]
#             if printnum < 5:
#                 print(m)
#                 printnum += 1
#     m.sort_index(axis=0, inplace=True)
#     m.sort_index(axis=1, inplace=True)
#     m.to_csv(INPUTPATH[:-4] + "_" + str(year) + ".csv")
#     # plot_heat_map(m, INPUTPATH[:-4] + "_" + str(year) + "_heat.pdf", INPUTPATH[:-4] + str(year))
#     print( "padded " + str(year))


    for year in range(STARTYEAR, STOPYEAR):
        seen_ciks = set()
        year_repr = []
        for i in M.columns:
            if "_" + str(year) in i and i != "7380_0000833444_1997_ADT-LIMITED" and i != "3559_0000006951_2004_APPLIED-MATERIALS-INC-/DE":
                year_repr.append(i)
                seen_ciks.add(i[5:15])
        m = M[year_repr].T[year_repr].T
        for cik in ciks_to_sics:
            if cik not in seen_ciks:
                newcolname = ciks_to_sics[cik] + "_" + cik + "_" + str(year) + "_missing"
                m[newcolname] = [np.nan for i in range(len(m.index))]
                mt = m.T
                mt[newcolname] = [np.nan for i in range(len(mt.index))]
                m = mt.T
                # m.append(pandas.DataFrame({k:[np.nan] for k in m.columns}, columns=m.columns))
                # if printnum < 5:
                #     print(m)
                #     printnum += 1
        m.sort_index(axis=0, inplace=True)
        m.sort_index(axis=1, inplace=True)
        # m.to_csv(INPUTPATH[:-4]+ "_" + str(year) + ".csv")
        m.to_csv(OUTPUTPATH + "_" + str(year) + ".csv")
        # m.to_csv(INPUTPATH[:, INPUTPATH.index("/") + 1] + "sims_by_year/" + INPUTPATH[INPUTPATH.index("/")+1:-4] + "_" + str(year) + ".csv")
        # plot_heat_map(m, INPUTPATH[:-4] + "_" + str(year) + "_heat.pdf", INPUTPATH[:-4] + str(year))
        print( "padded " + str(year))

# for year in range(STARTYEAR, STOPYEAR):
#     year_repr = [i for i in M.columns if "_" + str(year) in i]
#     m = M[year_repr].T[year_repr].T
#     for cik in ciks_to_sics:
#         if cik not in m.columns:
#             newcolname = ciks_to_sics[cik] + "_" + cik + "_" + str(year) + "_missing"
#             m.append(pandas.DataFrame([np.nan for k in m.columns], columns=m.columns, )
#                 {k:[np.nan] for k in m.columns}, ignore_index=True)
#             m[newcolname] = [np.nan for i in range(len(m.index))]
#             if printnum < 5:
#                 print(m)
#                 printnum += 1
#     m.sort_index(axis=0, inplace=True)
#     m.sort_index(axis=1, inplace=True)
#     m.to_csv(INPUTPATH[:-4] + "_" + str(year) + ".csv")
#     # plot_heat_map(m, INPUTPATH[:-4] + "_" + str(year) + "_heat.pdf", INPUTPATH[:-4] + str(year))
#     print( "padded " + str(year))

yearly_with_padding("SPv8_nouns_doc2vec_similarities.csv", "SPv8_nouns_doc2vec_similarities", 1994, 2019)
# yearly_with_padding("out.txt", 1994, 2019)