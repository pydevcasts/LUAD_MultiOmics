"""Curated gene lists for key cell death and cell cycle pathways.

Sources: KEGG, Reactome, MSigDB, and recent literature (2023-2025).
These pathways are directly relevant to cancer progression and survival.
"""

PATHWAY_GENE_SETS = {
    "mitophagy": [
        "PINK1", "PRKN", "PARK2", "BNIP3", "BNIP3L", "FUNDC1",
        "ULK1", "ULK2", "ULK3", "BECN1", "ATG5", "ATG7", "ATG12",
        "ATG13", "ATG14", "ATG16L1", "ATG2A", "ATG2B", "ATG3",
        "ATG4A", "ATG4B", "ATG4C", "ATG4D", "ATG9A", "ATG9B",
        "ATG101", "MAP1LC3A", "MAP1LC3B", "MAP1LC3C",
        "GABARAP", "GABARAPL1", "GABARAPL2",
        "SQSTM1", "OPTN", "CALCOCO2", "TAX1BP1", "NBR1",
        "VDAC1", "VDAC2", "VDAC3",
        "TOMM7", "TOMM20", "TOMM40", "TOMM22",
        "MFN1", "MFN2", "DNM1L", "FIS1", "MFF", "MIEF1", "MIEF2",
        "USP30", "USP15", "TBK1", "FKBP8", "AMBRA1",
        "TBC1D15", "TBC1D17", "RHEB", "MTOR",
        "PRKAA1", "PRKAA2", "TSC1", "TSC2",
        "HIF1A", "PHB2", "NIPSNAP1", "NIPSNAP2",
        "WIPI1", "WIPI2", "RB1CC1",
        "PIK3C3", "PIK3R4", "UVRAG", "RUBCN", "RUBCNL",
        "VPS34", "ATG10", "ATG16L2",
    ],

    "cuproptosis": [
        "FDX1", "FDX2", "LIAS", "LIPT1", "LIPT2",
        "DLD", "DLAT", "DLST", "PDHA1", "PDHB", "PDHX",
        "OGDH", "OGDHL", "SUCLG1", "SUCLA2", "SUCLG2",
        "SDHA", "SDHB", "SDHC", "SDHD", "SDHAF1", "SDHAF2",
        "FH", "IDH1", "IDH2", "IDH3A", "IDH3B", "IDH3G",
        "CS", "ACO1", "ACO2",
        "MTF1", "GLS", "GLS2",
        "SLC31A1", "ATP7A", "ATP7B",
        "DBT", "GCSH", "AMT",
        "CDKN2A", "SLC7A11",
    ],

    "ferroptosis": [
        "GPX4", "SLC7A11", "SLC3A2",
        "FTH1", "FTL", "SLC40A1", "TFRC", "TF",
        "ACSL4", "LPCAT3",
        "ALOX5", "ALOX12", "ALOX15", "ALOX15B", "ALOX12B",
        "NOX1", "NOX4", "NOX5", "CYBB", "CYBA", "NCF1", "NCF2", "NCF4",
        "NFE2L2", "KEAP1", "HMOX1", "NQO1",
        "GCLC", "GCLM", "GSS", "GSR",
        "SOD1", "SOD2", "CAT",
        "PRDX1", "PRDX2", "PRDX6",
        "TXN", "TXNRD1", "TXNRD2",
        "AIFM2", "DHODH", "GCH1",
        "MDM2", "MDM4", "TP53",
        "SAT1", "CDO1", "CBS", "CTH",
        "HSPB1", "DPP4",
        "VDAC2", "VDAC3",
        "CISD1", "CISD2",
        "BECN1", "ATG5", "ATG7",
        "FANCD2",
        "SLC1A5", "SLC6A9",
        "G6PD", "PGD", "TKT", "TALDO1", "RPIA",
        "RB1", "ZEB1", "YAP1", "WWTR1",
        "LATS1", "LATS2", "STK3", "STK4",
        "SLC38A1", "SLC38A2",
        "PCBD1", "SPR", "PTS", "GCHFR",
        "NFS1", "ISCU", "ISCA1", "ISCA2", "IBA57", "BOLA3", "NFU1", "IND1",
        "COQ2", "COQ3", "COQ4", "COQ5", "COQ6", "COQ7", "COQ8A", "COQ8B",
        "GPX1", "GPX2", "GPX3", "GPX6", "GPX7",
    ],

    "p53_signaling": [
        "TP53", "TP63", "TP73",
        "MDM2", "MDM4",
        "CDKN1A", "CDKN2A", "CDKN1B",
        "BAX", "BAK1", "BBC3", "PMAIP1",
        "BCL2", "BCL2L1", "MCL1",
        "BID", "BAD", "BCL2L11", "BMF", "HRK", "BLK", "BIK",
        "CASP3", "CASP6", "CASP7", "CASP8", "CASP9",
        "APAF1", "CYCS", "DIABLO", "HTRA2",
        "XIAP", "BIRC2", "BIRC3", "BIRC5",
        "FAS", "FASLG", "TNFRSF10A", "TNFRSF10B", "TNFSF10",
        "PERP", "PIDD1", "FADD", "TRADD", "RIPK1",
        "ATM", "ATR", "CHEK1", "CHEK2",
        "TP53BP1", "BRCA1", "BRCA2", "RAD51",
        "CCND1", "CCNE1", "CDK2", "CDK4", "CDK6",
        "RB1", "E2F1", "E2F2", "E2F3",
        "MYC", "TERT",
        "GADD45A", "GADD45B", "GADD45G",
        "SESN1", "SESN2", "SESN3",
        "TIGAR", "SCO2", "COX11",
        "GLS2",
        "PDK1", "PDK2", "PDK3", "PDK4",
        "LDHA", "LDHB",
        "HK1", "HK2", "PFKFB3",
        "PGAM1", "ENO1", "PKM", "GPI", "ALDOA", "TPI1", "GAPDH",
        "SIVA1", "EI24", "DRAM1", "TRAF4", "SERPINB5",
        "ZMAT3", "TNFRSF10D", "TNFRSF10C",
        "CASP1", "CASP2", "CASP10",
        "DDB2", "XPC", "ERCC1", "ERCC2", "ERCC3", "ERCC4", "ERCC5",
        "RRM2B", "STEAP3", "POLH", "POLI", "POLK", "REV1", "REV3L",
    ],

    "cell_cycle": [
        "CCNA1", "CCNA2", "CCNB1", "CCNB2", "CCNB3",
        "CCND1", "CCND2", "CCND3",
        "CCNE1", "CCNE2",
        "CCNF", "CCNG1", "CCNG2", "CCNH", "CCNI", "CCNK",
        "CCNT1", "CCNT2",
        "CDK1", "CDK2", "CDK3", "CDK4", "CDK6",
        "CDK7", "CDK8", "CDK9", "CDK10", "CDK11A", "CDK11B",
        "CDKN1A", "CDKN1B", "CDKN1C", "CDKN2A", "CDKN2B",
        "CDKN2C", "CDKN2D", "CDKN3",
        "RB1", "RBL1", "RBL2",
        "E2F1", "E2F2", "E2F3", "E2F4", "E2F5", "E2F6", "E2F7", "E2F8",
        "CDC6", "CDT1",
        "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7",
        "ORC1", "ORC2", "ORC3", "ORC4", "ORC5", "ORC6",
        "CDC25A", "CDC25B", "CDC25C",
        "WEE1", "WEE2", "MYT1",
        "PLK1", "PLK2", "PLK3", "PLK4",
        "AURKA", "AURKB", "AURKC",
        "BUB1", "BUB1B", "BUB3",
        "MAD1L1", "MAD2L1", "MAD2L2",
        "TTK", "CDC20", "FZR1",
        "ANAPC1", "ANAPC2", "ANAPC4", "ANAPC5", "ANAPC7",
        "ANAPC10", "ANAPC11", "ANAPC13",
        "CDC27", "CDC16", "CDC23", "CDC26",
        "SKP1", "SKP2", "CUL1", "RBX1",
        "FBXW7", "FBXO31", "FBXO4", "FBXO28",
        "CDC7", "DBF4",
        "TOPBP1", "CLASPIN", "TIMELESS", "TIPIN", "WDHD1",
        "STMN1", "KIF11", "KIF20A", "KIF23",
        "NUF2", "ZWINT", "CENPA", "CENPF", "CENPE",
        "MELK", "PBK", "BIRC5", "CKS1B", "CKS2",
        "PKMYT1", "NEK2", "NEK6", "NEK7",
        "TPX2", "CEP55", "ARL6IP1",
        "RRM1", "RRM2", "TYMS", "DHFR", "TK1",
        "TOP2A", "TOP1", "PCNA", "RFC1", "RFC2",
        "LIG1", "FEN1", "POLD1", "POLE", "POLA1", "PRIM1",
        "SMC1A", "SMC3", "RAD21", "STAG1", "STAG2",
        "ESPL1", "SGO1", "SGO2",
    ],

    "apoptosis": [
        "BAX", "BAK1", "BCL2", "BCL2L1", "BCL2L2", "MCL1",
        "BCL2A1", "BCL2L10", "BCL2L11", "BBC3", "PMAIP1",
        "BMF", "HRK", "BID", "BAD", "BIK", "BLK",
        "CASP1", "CASP2", "CASP3", "CASP4", "CASP5",
        "CASP6", "CASP7", "CASP8", "CASP9", "CASP10", "CASP14",
        "APAF1", "CYCS", "DIABLO", "HTRA2",
        "XIAP", "BIRC2", "BIRC3", "BIRC5", "BIRC6", "BIRC7", "BIRC8",
        "NAIP",
        "FAS", "FASLG", "TNFRSF1A", "TNFRSF1B",
        "TNFRSF10A", "TNFRSF10B", "TNFRSF10C", "TNFRSF10D",
        "TNFSF10", "TNF",
        "TRADD", "FADD", "RIPK1", "RIPK3",
        "CFLAR",
        "TRAF1", "TRAF2", "TRAF3", "TRAF5", "TRAF6",
        "IKBKB", "IKBKG", "CHUK",
        "NFKB1", "NFKB2", "RELA", "RELB", "REL",
        "BCL10", "MALT1", "CARD9", "CARD11",
        "PYCARD", "NLRP1", "NLRP3", "AIM2", "NLRC4",
        "GSDMD", "GSDME",
        "IL1A", "IL1B", "IL18", "IL33", "HMGB1",
        "DAXX", "MAP3K5", "MAP2K4", "MAP2K7",
        "MAPK8", "MAPK9", "MAPK10",
        "JUN", "FOS",
        "CASP8AP2", "RIPK2", "TRAF4",
        "LMNA", "LMNB1", "LMNB2",
        "PARP1", "PARP2",
        "DNASE1", "DNASE1L3", "DFFA", "DFFB",
        "ENDOG", "AIFM1",
        "TP53", "TP63", "TP73",
        "CDKN1A", "CDKN2A",
        "MDM2", "MDM4",
        "ATM", "ATR", "CHEK1", "CHEK2",
    ],
}


def get_all_pathway_genes():
    """Return union of all pathway genes."""
    all_genes = set()
    for pathway, genes in PATHWAY_GENE_SETS.items():
        all_genes.update(genes)
    return sorted(all_genes)


def get_pathway_gene_counts():
    """Return gene count per pathway."""
    return {k: len(v) for k, v in PATHWAY_GENE_SETS.items()}


def get_gene_to_pathways():
    """Return mapping from gene to list of pathways it belongs to."""
    gene_map = {}
    for pathway, genes in PATHWAY_GENE_SETS.items():
        for gene in genes:
            gene_map.setdefault(gene, []).append(pathway)
    return gene_map