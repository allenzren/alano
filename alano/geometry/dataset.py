def objectName2Ind(name):
    if name >= 200:
        return name - 200 + (60+11)
    elif name >= 100:
        return name - 100 + (60)
    else:
        return name


def getYCBInfo(objInd):
    # file name, use google or not, object position offset, object dimension, object yaw offset to align longer dim with y axis
    data = [
        ['000_rubiks_cube', 1, [0.057, 0.057, 0.057]],
        ['001_chips_can', 0, [0.0835*0.5, 0.0861*0.5, 0.2406*0.5]],
        ['002_master_chef_can', 1, [0.1025*0.5, 0.1024*0.5, 0.1402*0.5]],
        ['003_cracker_box', 0, [0.06*0.7, 0.158*0.7, 0.21*0.7]],
        ['004_sugar_box', 1, [0.038*0.7, 0.089*0.7, 0.175*0.7]],
        ['005_tomato_soup_can', 0, [0.066, 0.066, 0.101]],
        ['006_mustard_bottle', 0, [0.065*0.8, 0.0105*0.8, 0.185*0.8]],
        ['007_tuna_fish_can', 1, [0.0856*0.7, 0.0855*0.7, 0.0335*0.7]],
        ['008_pudding_box', 0, [0.035, 0.11, 0.089]],
        ['009_gelatin_box', 0, [0.028, 0.085, 0.073]],
        ['010_potted_meat_can', 0, [0.05, 0.097, 0.082]],
        ['011_banana', 1, [0.19*0.8, 0.19*0.8, 0.036*0.8]],
        ['012_strawberry', 1, [0.055*1.25, 0.055*1.25, 0.044*1.25]],
        ['013_apple', 1, [0.075, 0.075, 0.075]],
        ['014_lemon', 1, [0.054*1.25, 0.068*1.25, 0.068*1.25]],
        ['015_peach', 1, [0.059, 0.059, 0.059]],
        ['016_pear', 1, [0.06, 0.1, 0.066]],
        ['017_orange', 1, [0.073, 0.073, 0.073]],
        ['018_plum', 1, [0.052*1.25, 0.052*1.25, 0.052*1.25]],
        ['019_pitcher_base', 0, [0.1433*0.6, 0.1444*0.6, 0.2377*0.6]],
        ['020_bleach_cleanser', 1, [0.068, 0.103, 0.25*0.5]],
        ['021_bowl', 0, [0.16, 0.16, 0.053]],
        ['022_mug', 1, [0.09, 0.1176, 0.082]],
        ['023_skillet', 1, [0.2678*0.5, 0.4421*0.5, 0.1427*0.5]],
        ['024_plate', 1, [0.258, 0.258, 0.024]],
        ['025_fork', 1, [0.014*1.5, 0.198, 0.02*2]],
        ['026_spoon', 1, [0.014*2, 0.195, 0.014*1.5]],
        ['027_knife', 1, [0.014*1.5, 0.215, 0.02*1.5]],
        ['028_spatula', 1, [0.035*0.5, 0.35*0.5, 0.083*0.5]],
        ['029_power_drill', 1, [0.1878*0.8, 0.1878*0.8, 0.0577*0.8]],
        ['030_scissors', 1, [0.087*1.5, 0.2, 0.014*2]],
        ['031_padlock', 1, [0.047, 0.065*2, 0.024*2]],
        ['032_large_marker', 1, [0.018*1.2, 0.121*1.2, 0.018*1.2]],
        ['033_adjustable_wrench', 1, [0.055*2, 0.205, 0.005*4]],
        ['034_phillips_screwdriver', 1,  [0.031, 0.215*0.9, 0.031]],
        ['035_flat_screwdriver', 1, [0.031, 0.215*0.9, 0.031]],
        ['036_hammer', 1, [0.1324*0.6, 0.3309*0.5, 0.0332]],
        ['037_medium_clamp', 1, [0.078, 0.085, 0.027]],
        ['038_large_clamp', 1, [0.125, 0.165, 0.032]],
        ['039_extra_large_clamp', 1, [0.165*0.9, 0.213*0.9, 0.037*0.9]],
        ['040_mini_soccer_ball', 0, [0.1348*0.5, 0.1325*0.5, 0.1208*0.5]],
        ['041_softball', 0, [0.1017*0.6, 0.1057*0.6, 0.0913*0.8]],
        ['042_baseball', 1, [0.073*0.8, 0.073*0.8, 0.073*0.8]],
        ['043_tennis_ball', 1, [0.065, 0.065, 0.065]],
        ['044_golf_ball', 1, [0.043*1.5, 0.043*1.5, 0.043*1.5]],
        ['045_foam_brick', 1, [0.05, 0.075, 0.05]],
        ['046_a_cups', 0, [0.055*1.5, 0.055, 0.06]],
        ['047_b_cups', 0, [0.06, 0.06, 0.062]],
        ['048_c_cups', 0, [0.065*2, 0.065, 0.064]],
        ['049_d_cups', 0, [0.07*1.5, 0.07*1.5, 0.066*1.5]],
        ['050_e_cups', 0, [0.075*1.5, 0.075*1.5, 0.068]],
        ['051_f_cups', 0, [0.08*0.8, 0.08*0.8, 0.07*1.5]],
        ['052_b_toy_airplane', 1, [0.12, 0.18, 0.06]],
        ['053_c_toy_airplane', 1, [0.031*2, 0.067*2, 0.031*2]],
        ['054_d_toy_airplane', 1, [0.031*1.5, 0.031*1.5, 0.067*1.5]],
        ['055_a_lego_duplo', 1, [0.032*2, 0.064*2, 0.024*3]],
        ['056_b_lego_duplo', 0, [0.044, 0.06*2, 0.038*2]],
        ['057_c_lego_duplo', 1, [0.033*2, 0.064, 0.024*2]],
        ['058_d_lego_duplo', 1, [0.033, 0.048*3.5, 0.043*2]],
        ['059_e_lego_duplo', 1, [0.032, 0.096, 0.043]],
        ]

    return data[objInd][0], data[objInd][1], data[objInd][2]


def get3DNetInfo(objInd):
    # file name, object dimension
    data = [
        ['100_Mug', [0.0301*3, 0.0436*3, 0.0352*3]], # decom
        ['101_NeedleNose', [0.0450*2, 0.1174*1.6, 0.0063*5]], # decom
        ['102_Plier', [0.0450*3, 0.0535*3, 0.0067*5]], # decom
        ['103_Gooseneck', [0.0450*1.5, 0.1725, 0.0106*3]], # decom
        ['104_PlierStandard', [0.0450*2.5, 0.1150, 0.0083*3]], # decom
        ['105_RubberMallet', [0.0450, 0.1524, 0.0310*2]], # decom
        ['106_Screwdriver', [0.0358*1.2, 0.2439*0.8, 0.0450*1.2]], # decom
        ['107_SledgeHammer', [0.0450, 0.1451, 0.0156*3]], # decom
        ['108_SqrBowl', [0.0319*4, 0.0319*4, 0.0173*4]], # decom
        ['109_Ketchup', [0.0450, 0.0450, 0.1488]],  # decom
        ['110_Tetrabrik', [0.0304, 0.0450, 0.0775]],
        # ['111_Milk', [0.0450, 0.0450, 0.1138]]
        ]

    return data[objInd][0], data[objInd][1]


def getKITInfo(objInd):
    # file name, object dimension
    data = [
        ['200_Amicelli', [0.0452*1.5, 0.0401*1.5, 0.0741*1.5]],
        ['201_BathDetergent', [0.0435, 0.0423, 0.1308]],
        ['202_CondensedMilk', [0.0433*1.5, 0.0451*1.5, 0.056*1.5]],
        ['203_Curry', [0.0432, 0.0450, 0.1193]],
        ['204_DanishHam', [0.0450, 0.0642*2, 0.0234*2]],
        ['205_FizzyTablets', [0.0450*0.8, 0.2183*0.8, 0.0446*0.8]],
        ['206_GlassCup', [0.0429*3, 0.0496*3, 0.0309*3]],  # decom
        ['207_HamburgerSauce', [0.0283*2, 0.0450*2, 0.0924*1.5]],  # decom
        ['208_HeringTin', [0.0450, 0.0801*2, 0.0152*2]],
        ['209_InstantSoup', [0.0297, 0.0590*3, 0.0394*1.5]],
        ['210_LetterP', [0.0609*0.8, 0.0450*2, 0.0116*3]],  # decom
        ['211_Margarine', [0.0450*0.8, 0.0662*2, 0.0271]],
        ['212_Moon', [0.0450*2, 0.0564*2, 0.0089*6]],  # decom
        ['213_NutellaGo', [0.0213*2, 0.0392*3, 0.0375*3]],
        ['214_Sprayflask', [0.0281*1.5, 0.0450*1.5, 0.1143*1.5]],  # decom
        ['215_Sprudelflasche', [0.0448, 0.0450, 0.1636]],  # decom
        ['216_Waterglass', [0.0450*2, 0.0445*2, 0.0716*2]],  # decom
        ['217_BakingSoda', [0.0392*1.5, 0.0405*1.5, 0.0657*1.5]],
        ['218_ChocolateBars', [0.0179*2, 0.0377*2, 0.0561*2]],
        ['219_CoffeeBox', [0.0298*2, 0.0287*2, 0.0290*2]],
        ['220_Deodorant', [0.0450, 0.0450, 0.0949]],
        ['221_CoffeeCookies', [0.0450, 0.1487, 0.0432]],
        ['222_Shampoo', [0.0279, 0.0452, 0.1199]],
        ['223_Toothpaste', [0.0450, 0.1241, 0.0311]],
        ]

    return data[objInd][0], data[objInd][1]

def getAllInfo(objInd):
    # file name, use google or not, object position offset, object dimension, object yaw offset to align longer dim with y axis
    data = [
        ['000_rubiks_cube', 1, [0.057, 0.057, 0.057]],
        ['001_chips_can', 0, [0.0835*0.5, 0.0861*0.5, 0.2406*0.5]],
        ['002_master_chef_can', 1, [0.1025*0.5, 0.1024*0.5, 0.1402*0.5]],
        ['003_cracker_box', 0, [0.06*0.7, 0.158*0.7, 0.21*0.7]],
        ['004_sugar_box', 1, [0.038*0.7, 0.089*0.7, 0.175*0.7]],
        ['005_tomato_soup_can', 0, [0.066, 0.066, 0.101]],
        ['006_mustard_bottle', 0, [0.065*0.8, 0.0105*0.8, 0.185*0.8]],
        ['007_tuna_fish_can', 1, [0.0856*0.7, 0.0855*0.7, 0.0335*0.7]],
        ['008_pudding_box', 0, [0.035, 0.11, 0.089]],
        ['009_gelatin_box', 0, [0.028, 0.085, 0.073]],
        ['010_potted_meat_can', 0, [0.05, 0.097, 0.082]],
        ['011_banana', 1, [0.19*0.8, 0.19*0.8, 0.036*0.8]],
        ['012_strawberry', 1, [0.055*1.25, 0.055*1.25, 0.044*1.25]],
        ['013_apple', 1, [0.075, 0.075, 0.075]],
        ['014_lemon', 1, [0.054*1.25, 0.068*1.25, 0.068*1.25]],
        ['015_peach', 1, [0.059, 0.059, 0.059]],
        ['016_pear', 1, [0.06, 0.1, 0.066]],
        ['017_orange', 1, [0.073, 0.073, 0.073]],
        ['018_plum', 1, [0.052*1.25, 0.052*1.25, 0.052*1.25]],
        ['019_pitcher_base', 0, [0.1433*0.6, 0.1444*0.6, 0.2377*0.6]],
        ['020_bleach_cleanser', 1, [0.068, 0.103, 0.25*0.5]],
        ['021_bowl', 0, [0.16, 0.16, 0.053]],
        ['022_mug', 1, [0.09, 0.1176, 0.082]],
        ['023_skillet', 1, [0.2678*0.5, 0.4421*0.5, 0.1427*0.5]],
        ['024_plate', 1, [0.258, 0.258, 0.024]],
        ['025_fork', 1, [0.014*1.5, 0.198, 0.02*2]],
        ['026_spoon', 1, [0.014*2, 0.195, 0.014*1.5]],
        ['027_knife', 1, [0.014*1.5, 0.215, 0.02*1.5]],
        ['028_spatula', 1, [0.035*0.5, 0.35*0.5, 0.083*0.5]],
        ['029_power_drill', 1, [0.1878*0.8, 0.1878*0.8, 0.0577*0.8]],
        ['030_scissors', 1, [0.087*1.5, 0.2, 0.014*2]],
        ['031_padlock', 1, [0.047, 0.065*2, 0.024*2]],
        ['032_large_marker', 1, [0.018*1.2, 0.121*1.2, 0.018*1.2]],
        ['033_adjustable_wrench', 1, [0.055*2, 0.205, 0.005*4]],
        ['034_phillips_screwdriver', 1,  [0.031, 0.215*0.9, 0.031]],
        ['035_flat_screwdriver', 1, [0.031, 0.215*0.9, 0.031]],
        ['036_hammer', 1, [0.1324*0.6, 0.3309*0.5, 0.0332]],
        ['037_medium_clamp', 1, [0.078, 0.085, 0.027]],
        ['038_large_clamp', 1, [0.125, 0.165, 0.032]],
        ['039_extra_large_clamp', 1, [0.165*0.9, 0.213*0.9, 0.037*0.9]],
        ['040_mini_soccer_ball', 0, [0.1348*0.5, 0.1325*0.5, 0.1208*0.5]],
        ['041_softball', 0, [0.1017*0.6, 0.1057*0.6, 0.0913*0.8]],
        ['042_baseball', 1, [0.073*0.8, 0.073*0.8, 0.073*0.8]],
        ['043_tennis_ball', 1, [0.065, 0.065, 0.065]],
        ['044_golf_ball', 1, [0.043*1.5, 0.043*1.5, 0.043*1.5]],
        ['045_foam_brick', 1, [0.05, 0.075, 0.05]],
        ['046_a_cups', 0, [0.055*1.5, 0.055, 0.06]],
        ['047_b_cups', 0, [0.06, 0.06, 0.062]],
        ['048_c_cups', 0, [0.065*2, 0.065, 0.064]],
        ['049_d_cups', 0, [0.07*1.5, 0.07*1.5, 0.066*1.5]],
        ['050_e_cups', 0, [0.075*1.5, 0.075*1.5, 0.068]],
        ['051_f_cups', 0, [0.08*0.8, 0.08*0.8, 0.07*1.5]],
        ['052_b_toy_airplane', 1, [0.12, 0.18, 0.06]],
        ['053_c_toy_airplane', 1, [0.031*2, 0.067*2, 0.031*2]],
        ['054_d_toy_airplane', 1, [0.031*1.5, 0.031*1.5, 0.067*1.5]],
        ['055_a_lego_duplo', 1, [0.032*2, 0.064*2, 0.024*3]],
        ['056_b_lego_duplo', 0, [0.044, 0.06*2, 0.038*2]],
        ['057_c_lego_duplo', 1, [0.033*2, 0.064, 0.024*2]],
        ['058_d_lego_duplo', 1, [0.033, 0.048*3.5, 0.043*2]],
        ['059_e_lego_duplo', 1, [0.032, 0.096, 0.043]],

        ['100_Mug', [0.0301*3, 0.0436*3, 0.0352*3]], # decom
        ['101_NeedleNose', [0.0450*2, 0.1174*1.6, 0.0063*5]], # decom
        ['102_Plier', [0.0450*3, 0.0535*3, 0.0067*5]], # decom
        ['103_Gooseneck', [0.0450*1.5, 0.1725, 0.0106*3]], # decom
        ['104_PlierStandard', [0.0450*2.5, 0.1150, 0.0083*3]], # decom
        ['105_RubberMallet', [0.0450, 0.1524, 0.0310*2]], # decom
        ['106_Screwdriver', [0.0358*1.2, 0.2439*0.8, 0.0450*1.2]], # decom
        ['107_SledgeHammer', [0.0450, 0.1451, 0.0156*3]], # decom
        ['108_SqrBowl', [0.0319*4, 0.0319*4, 0.0173*4]], # decom
        ['109_Ketchup', [0.0450, 0.0450, 0.1488]],  # decom
        ['110_Tetrabrik', [0.0304, 0.0450, 0.0775]],

        ['200_Amicelli', [0.0452*1.5, 0.0401*1.5, 0.0741*1.5]],
        ['201_BathDetergent', [0.0435, 0.0423, 0.1308]],
        ['202_CondensedMilk', [0.0433*1.5, 0.0451*1.5, 0.056*1.5]],
        ['203_Curry', [0.0432, 0.0450, 0.1193]],
        ['204_DanishHam', [0.0450, 0.0642*2, 0.0234*2]],
        ['205_FizzyTablets', [0.0450*0.8, 0.2183*0.8, 0.0446*0.8]],
        ['206_GlassCup', [0.0429*3, 0.0496*3, 0.0309*3]],  # decom
        ['207_HamburgerSauce', [0.0283*2, 0.0450*2, 0.0924*1.5]],  # decom
        ['208_HeringTin', [0.0450, 0.0801*2, 0.0152*2]],
        ['209_InstantSoup', [0.0297, 0.0590*3, 0.0394*1.5]],
        ['210_LetterP', [0.0609*0.8, 0.0450*2, 0.0116*3]],  # decom
        ['211_Margarine', [0.0450*0.8, 0.0662*2, 0.0271]],
        ['212_Moon', [0.0450*2, 0.0564*2, 0.0089*6]],  # decom
        ['213_NutellaGo', [0.0213*2, 0.0392*3, 0.0375*3]],
        ['214_Sprayflask', [0.0281*1.5, 0.0450*1.5, 0.1143*1.5]],  # decom
        ['215_Sprudelflasche', [0.0448, 0.0450, 0.1636]],  # decom
        ['216_Waterglass', [0.0450*2, 0.0445*2, 0.0716*2]],  # decom
        ['217_BakingSoda', [0.0392*1.5, 0.0405*1.5, 0.0657*1.5]],
        ['218_ChocolateBars', [0.0179*2, 0.0377*2, 0.0561*2]],
        ['219_CoffeeBox', [0.0298*2, 0.0287*2, 0.0290*2]],
        ['220_Deodorant', [0.0450, 0.0450, 0.0949]],
        ['221_CoffeeCookies', [0.0450, 0.1487, 0.0432]],
        ['222_Shampoo', [0.0279, 0.0452, 0.1199]],
        ['223_Toothpaste', [0.0450, 0.1241, 0.0311]],
        ]

    return data[objInd][-1]

    # ['201_bakingSoda', [0.0392, 0.0405, 0.0657]],
    # ['202_ChocoIcing', [0.0468, 0.0423, 0.0343]],
    # ['202_ChocolateBars', [0.0179, 0.0377, 0.0561]],
    # ['205_CoffeeBox', [0.0298, 0.0287, 0.0290]],
    # ['206_CoffeeCookies', [0.0450, 0.1487, 0.0432]],
    # ['202_CokePlasticLarge', [0.0450, 0.0448, 0.1635]],
    # ['205_Deodorant', [0.0450, 0.0448, 0.0949]],
    # ['206_FlowerCup', [0.0314, 0.0436, 0.0463]],  # decom
    # ['206_Glassbowl', [0.0452, 0.0450, 0.0174]],  # decom
    # ['208_Heart1', [0.0450*2, 0.0636*2, 0.0261*2]],  # decom
    # ['214_Pitcher', [0.0280*4, 0.0459*4, 0.0366*4]],  # decom
    # ['214_RedCup', [0.0366*3, 0.0480*3, 0.0410*3]],  # decom
    # ['214_Shampoo', [0.0278, 0.0450, 0.1197]],  # decom
    # ['214_SmallGlass', [0.0435, 0.0450, 0.0629]],  # decom
    # ['216_Toothpaste', [0.0311, 0.0450, 0.1241], 2],
    # ['217_Wineglass', [0.0450, 0.0449, 0.0994], 2]  # decom