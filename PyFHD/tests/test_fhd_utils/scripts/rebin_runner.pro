PRO rebin_runner
    ;Setup the data for the tests
    test = [0,10,20,30]
    test2 = [[0], [10],[20],[30]]
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
    
    ;Do the Tests
    
    
    SAVE, /VARIABLES, FILENAME = 'rebin.sav'
END