use opencv::core::{bitwise_not, no_array, Rect, Scalar, Vec3b, CV_8UC1, CV_8UC3, CV_8UC4};

use opencv::highgui::{imshow, wait_key};
use opencv::prelude::*;

fn main() -> opencv::Result<()> {
    // let mat = Mat::new_rows_cols_with_default(200, 200, CV_8UC1, 0.into())?;
    // Mat::roi(&mat, Rect::new(50, 50, 100, 100))?.set_scalar(Scalar::from(255))?;

    let mat = Mat::new_rows_cols_with_default(200, 200, CV_8UC1, 0.into())?;

    Mat::roi(&mat, Rect::new(50, 50, 100, 100))?.set_scalar(Scalar::from(255))?;

    let mut ddd = Mat::default();

    bitwise_not(&mat, &mut ddd, &no_array()).unwrap();

    imshow("adasd", &ddd).unwrap();

    wait_key(10000).unwrap();

    Ok(())
}
