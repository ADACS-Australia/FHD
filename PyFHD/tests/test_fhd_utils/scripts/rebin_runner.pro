PRO rebin_runner
    ;Setup the data for the tests
    test = [0,10,20,30]
    test2 = [[0],[10],[20],[30]]
    data = [[ -5,   4,   2,  -8,   1],$
            [  3,   0,   5,  -5,   1],$
            [  6,  -7,   4,  -4,  -8],$
            [ -1,  -5, -14,   2,   1]]
    data2 = [[ -5,   4,   2,  -8,   1,  4],$
             [  3,   0,   5,  -5,   1,  4],$
             [  6,  -7,   4,  -4,  -8,  3],$
             [ -1,  -5, -14,   2,   1,  8]]
    data3 = [[  5, -4,  8,  0],$
             [  9, 10, 20,  2],$
             [  1,  0,  1,  3],$
             [ 15, -12,  5, 4]]
    hundred = FLOOR(ABS(10*RANDOMU(42, 10, 10)))
    billion = FLOOR(ABS(10*RANDOMU(42,1e4, 1e5)))
    
    ;Do the Tests
    test_1R_8C = rebin(test, 8, 1)

    test_2R_8C = rebin(test, 8, 2)

    test_2C_1R = rebin(test, 2, 1)

    test_2R_2C = rebin(test, 2, 2)

    test_1R_1C = rebin(test, 1, 1)

    test_same = rebin(test, 4, 1)

    test2_vertical = rebin(test2, 1, 8)

    test2_to_square = rebin(test2, 4, 4)
    
    test2_to_smaller_square = rebin(test2, 2, 2)

    test2_to_rect = rebin(test2, 4, 8)

    test_to_smaller_rect = rebin(test, 1, 2)

    test2_same = rebin(test2, 1, 4)

    data_4R_10C = rebin(data, 10, 4)

    data_4R_15C = rebin(data, 15, 4)

    data_8R_10C = rebin(data, 10, 8)

    data_12R_10C = rebin(data, 10, 12)

    data_8R_15C = rebin(data, 15, 8)

    data_same = rebin(data, 5, 4)

    data2_2R_3C = rebin(data2, 3, 2)

    data2_2R_2C = rebin(data2, 2, 2)

    data3_2R_2C = rebin(data3, 2, 2)

    data3_1R_1C = rebin(data3, 1, 1)

    ;Larger size tests

    hundred_100R_100C = rebin(hundred, 100, 100)

    hundred_1KR_1KC = rebin(hundred, 1000, 1000)

    hundred_1E4R_1E5C = rebin(hundred, 1e5, 1e4)

    billion_100R_100C = rebin(billion, 100, 100)

    billion_1KR_1KC = rebin(billion, 1000, 1000)
    
    ;Adjust the path as required on the system you're running
    SAVE, /VARIABLES, FILENAME = 'rebin.sav'
END