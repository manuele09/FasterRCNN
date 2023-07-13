#The label name for each target
str_label = ["Head", "Helmet", "Welding Mask",
                          "Ear Protection", "Chest", "HVV", "Person"]

#The color to use for the bounding of each target
colors_bounding = ['red', 'blue', 'green',
                       'orange', 'purple', 'pink', "brown"]

# f"https://studentiunict-my.sharepoint.com/:u:/g/personal/cnnmnl01r09m088w_studium_unict_it/ESRH0_-lm7BKr4jRUZRyhQgBlStl96JNnQ1mpKn2mhXw8g?download=1"
virtual_dataset_train = {"train.virtual.txt": "ESRH0_-lm7BKr4jRUZRyhQgBlStl96JNnQ1mpKn2mhXw8g",                         
                         "group_0": "ERf726zSyetKpEMBoCgER98BpjOC1DCFxG4JweGMckKRKA",
                         "group_1": "EXJ8NAapu-1OpMtLrZHB2nYBeA4ihaj25MkqP7_X5cU0aQ",
                         "group_2": "EYr2Vm3cTHVPn71sCXVYh-oB4kwfPLf3h92xksdRWSkl5A",
                         "group_3": "ES3TDB2YyT5MrntnsXOb6lYBh9jsnYLrG1eCvsTC5H-_ZA",
                         "group_4": "EQF1h-XDbpFIkbY85z-atTcBahhuLLh6m72qSkwIPqjSNQ",
                         "group_5": "EbNIPDSnqmtGsOgEdZIG3CIBKuctGxA43hPaX4R4EmZP7A",
                         "group_6": "EUHy3XNiFEZFpzjWxKyd6RQBy_iMstYTpId550fw4CBvHA",
                         "group_7": "ERiFzJOn7hRFv8MiEXGRqqcBIkOIpgd9GKC0RRZfQZCUzw",
                         "group_8": "EandlODyoGROn6CxM_luD1UBBmHnCL4QqG3Qz4IGxYffeQ",
                         "group_9": "EROuH231Jg1AudGXiA4bSqUBsATbx7rACNtSHgC-o_rsGQ",
                         "group_10": "EUT3kjz--2xIjA0DM-qe8OIBRAfnbFhBtxw4UWNfL8EElA",
                         "group_11": "ERqlsvhQpYtJup8A-EVL3w0BnU591XnXHAo4yj360K7UGw",
                         "group_12": "EetY8AztGlBFhlDKyj3kTuYB-g_HiDqUh8J5G7i_nsG7cQ",
                         "group_13": "EeoWWxmf8yFMrfyKlz2JrfYBxK0jKp_VkJKb-eyNHPQ43Q",
                         "group_14": "EUj7sdgMZ2hDvuagyX4AYqUBlA0uA8hm0FhUEMGHDIo-ZA",
                         "group_15": "EfuGZjOJKLxOtgs1iJ8vyXABSwRkz8KCdO994NEYv48ypg",
                         "group_16": "Ee2G-cGNgUBFr13PUZTOkKYB07REnEJDK8lC4Cnx_sQljg",
                         "group_17": "EfaA32jOwR5IopCNiP_EH4kBGjUvViNgJ04fls6CS79EcQ",
                         "group_18": "Eecun1vLi-VChu3tOFXjNAoBkTyvlMzieELscP9osdeUUg",
                         "group_19": "ES4WnpNHHTxPrtsQvtHuqj8BKHJqzKJDmBY-In0ehWThDg"}

virtual_dataset_valid = {"valid.virtual.txt": "ESV7PFigkP5EicRmdChtQdMBjBTonHCgEM0-OHhk4i3phQ",
                         "group_20": "EZ2KJoTua-1LvaR74opw2YoB-G6OlNUPJKVFNPsKGp_X_g"}
  
real_dataset_train = {"train.real.txt": "EUG7D7Go-apBr9uT-BSQ3m0BA6in820suR6ZeQjM8D25MA",
                                   "train": "EblpJ8WwjddEgGDCm49BNf0B8flwHM0jJqimCT0ovwpBAQ"}

real_dataset_valid = {"valid.real.txt": "EduThXcR-TtJnKe_n0NQ5TQB1anZohLn93jYG-WsuH4Mag",
                                   "valid": "EVHyT5P7QAFDljbnhNWlotQBEdnTPennUtvQisiitKl_Og"}





 # self.dirs_ids = {"train.virtual.txt": "ESRgAfYQkchGj4Hjfl_lZLMBoLNTrhkHwPJzYBGsrt4SeA",
        #                  "valid.virtual.txt": "EXRzg_URR-ZEoYMpYH6W8R4BJWUVq4HMKZe-bvoq8ngUXw",
        #                  "27_03_19_19_15_32": "EVujRKjyKSJDiQ_8b-46r7sBSoY7yMre_UiHVXy4W3c14w",
        #                  "27_03_19_19_47_44": "EVTdTdDHT6FPkTAh-zK2JaoBQFNXpsHJfiKtOlxlga5dDQ",
        #                  "27_03_19_20_16_23":  "EZnnM9VW7qxCpuOoMd3DD70BTxf3qTUSzSFo1ItAcpzvVQ",
        #                  "27_03_19_20_43_51": "EbFMsp8MSwlDgXkS0EguZkwBhw5DCgi2nO3yTtjl426WMQ",
        #                  "27_03_19_21_09_00": "EUTcg4dh9G1Ji1QSQ34MTIIBBavjnOQYRAA0RsEx_Z4VqA",
        #                  "29_03_19_03_39_42": "ESDmLETIsShNlE2oEJI1D2QB2trOJCjWsJsc3O-dssu_ag",
        #                  "29_03_19_04_06_35": "EdoGdVHUsLRJiMxh0a1VBBEBrYeC2eX0Elvs1Jhq_b2gmw",
        #                  "29_03_19_04_34_54": "Ed4ZNXeY9a9HgY4T6MJJ7f0ByGK1EwbQmTxI8m6ijxDiBQ",
        #                  "29_03_19_05_01_34": "Ee9SYf6BuCxJiCpNbdXMTxoBEVHGA66aSHfsdmS_leHswQ",
        #                  "29_03_19_05_27_16": "Ed5KYH5YpX9BuS-5L4XDI9UBslYiXpRFs_UR8G_lBeQZzA",
        #                  "29_03_19_05_51_06": "EUi2CAMAvChJhPgQ0WNZGSoBGkuF0RQtD-4JXkhdJgNrEg",
        #                  "29_03_19_06_18_10": "EVnFqP7dx4VDnQ8XhiCa3W8B7VnESrnCYDghjrfGWU6xYQ",
        #                  "29_03_19_06_49_23": "EZI6BtKNLKNPqvXrH1WV-yQBX8KE-aAYfxaYqAZwT2048A",
        #                  "29_03_19_07_14_43": "EcrxUMpbayNLgMFvdH99rcQBCPedcn6QKavKeecMAPOGDA",
        #                  "29_03_19_07_45_24": "ERFr_pA4MRRIiemPRdMcoJwBuRjXdg62UYsgm9NAR1dDOA",
        #                  "29_03_19_08_12_28": "EWltsNr9UHdJsiC88YFQdmoB76AwtIFy6wea4oHZMRCNTw",
        #                  "29_03_19_08_39_27": "EaQFe7l04HxMpqzaYsxxQdUBLAtAKfESjI5jHgOg8Yz8PQ",
        #                  "29_03_19_09_05_50": "EZC6nbmQuz5OjFBKaTIoeRMBDSrtW6bNG-HNbB-F8DV3_Q",
        #                  "29_03_19_11_23_30": "EdDkwmpyxRpBqyCRbW3_75ABg4rPucqeMs-3afhEkEE7fA",
        #                  "29_03_19_11_48_52": "ERkF1A2H8NZPrIXx2EIZcyoBw10Q9k2QG2gIzvmMnxUXTA",
        #                  "29_03_19_12_11_24": "Ec51v9AKNTRPo1DI-YFLkrwBZptCy3XcPG-zZXHGHPYwsA",
        #                  "29_03_19_12_36_41": "ET5MAzxCLdRApgu-Yd4QSIsBIUIK2ZO1gc5VCgPpyC2hVw",
        #                  "29_03_19_13_16_22": "Eb6aTsYREItFrJb0kcbIbYEBWNHdlihTP7GTKOdgXC18Hg",
        #                  "29_03_19_13_40_24": "Ec8f5L291nBDkB6Z8XdbuZQB5fzAu9Rvb_3b0j8331ihNg",
        #                  "29_03_19_14_05_16": "EW0WMufbzspGim2ktl1jRosBwgeVU351rGaNTe4uy4VgRw",
        #                  "29_03_19_14_34_53": "EShJyt7_ELVOjxxmje0tYK8BImM7XYIlnLXaBZLm0f5iCQ",
        #                  "29_03_19_15_07_02": "EfjQr--s7u1NrkA13UNBZisBk596wqeFYFAr9qshRuefsw",
        #                  "29_03_19_15_39_29": "EXxy-jkcuLtHpR5vWX2y3ukBctruwvFqO9KNb2PshLSuow",
        #                  "29_03_19_16_06_55": "EQ1oEyDMhs9Au2AtS9wyIU4BnWKFxAg6UJyL8oD8I-8y2w",
        #                  "29_03_19_16_29_52": "EdKf-fzUaxxPk8wE_lykl2wBkec_JR4gjEtXpjdzuIl3Qw"}
