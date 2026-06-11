BOARD_COLS      = 9      # inner corners along width  (10 squares - 1)
BOARD_ROWS      = 14     # inner corners along height (15 squares - 1)
SQUARE_SIZE_MM  = 25.0   # physical size of one square in mm

CAMERA_WIDTH    = 1280
CAMERA_HEIGHT   = 720
CAMERA_FPS      = 30

IMAGES_DIR      = "images"        # parent; left/ and right/ live inside
LEFT_DIR        = "images/left"
RIGHT_DIR       = "images/right"
DEPTH_DIR       = "depth"
RESULTS_DIR     = "results"

MIN_CALIB_IMAGES = 20    # warn if fewer valid images
