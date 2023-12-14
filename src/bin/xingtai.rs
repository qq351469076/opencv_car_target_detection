use opencv::core::{Mat, Point, Scalar, Size, BORDER_CONSTANT, CV_8U};
use opencv::highgui::{imshow, wait_key};
use opencv::imgcodecs::{imread, IMREAD_COLOR};
use opencv::imgproc::{
    adaptive_threshold, canny, cvt_color, dilate, erode, get_structuring_element, morphology_ex,
    threshold, ADAPTIVE_THRESH_GAUSSIAN_C, COLOR_BGR2GRAY, MORPH_BLACKHAT, MORPH_GRADIENT,
    MORPH_RECT, MORPH_TOPHAT, THRESH_BINARY, THRESH_BINARY_INV,
};
use opencv::prelude::*;

/// 二值化
fn erzhihua() -> opencv::Result<()> {
    let mut new_mat = Mat::default();

    let raw_mat = imread("C:\\Users\\Administrator\\Desktop\\1.png", IMREAD_COLOR)?;

    // 转成灰度图
    cvt_color(&raw_mat, &mut new_mat, COLOR_BGR2GRAY, 0)?;

    let mut new_mat_2 = Mat::default();
    threshold(&new_mat, &mut new_mat_2, 100f64, 200f64, THRESH_BINARY)?;

    imshow("asdsa3", &new_mat_2)?;

    wait_key(10000)?;

    Ok(())
}

/// 自适应阈值
///
///由于光照不均匀以及阴影存在, 只有一个阈值会使得在阴影处的白色被二值化成黑色
fn zishiying() -> opencv::Result<()> {
    let mut new_mat = Mat::default();

    let raw_mat = imread("C:\\Users\\Administrator\\Desktop\\1.png", IMREAD_COLOR)?;

    // 转成灰度图
    cvt_color(&raw_mat, &mut new_mat, COLOR_BGR2GRAY, 0)?;

    let mut new_mat_2 = Mat::default();
    adaptive_threshold(
        &new_mat,
        &mut new_mat_2,
        255f64,
        ADAPTIVE_THRESH_GAUSSIAN_C,
        THRESH_BINARY,
        3,
        0f64,
    )?;

    imshow("asdsa3", &new_mat_2)?;

    wait_key(10000)?;

    Ok(())
}

/// 腐蚀
///
/// 将图片进行瘦身, 卷积核越大, 腐蚀越明显
fn fushi() -> opencv::Result<()> {
    let raw_mat = imread("C:\\Users\\Administrator\\Desktop\\j.png", IMREAD_COLOR)?;

    let size = Mat::new_rows_cols_with_default(3, 3, CV_8U, Scalar::from(1))?;

    let mut new_mat_2 = Mat::default();
    erode(
        &raw_mat,
        &mut new_mat_2,
        &size,
        Point::new(-1, -1),
        1,
        BORDER_CONSTANT,
        Scalar::from(0),
    )?;

    imshow("asdsa3", &new_mat_2)?;

    wait_key(10000)?;

    imshow("asdsa23", &raw_mat)?;

    wait_key(10000)?;

    Ok(())
}

/// 用现成的卷积核
fn juanjihe() -> opencv::Result<()> {
    let m = get_structuring_element(MORPH_RECT, Size::new(7, 7), Point::new(-1, -1))?;

    let raw_mat = imread("C:\\Users\\Administrator\\Desktop\\j.png", IMREAD_COLOR)?;

    let mut new_mat_2 = Mat::default();
    dilate(
        &raw_mat,
        &mut new_mat_2,
        &m,
        Point::new(-1, -1),
        1,
        BORDER_CONSTANT,
        Scalar::from(0),
    )?;

    imshow("asdsa3", &new_mat_2)?;

    wait_key(10000)?;

    imshow("asdsa23", &raw_mat)?;

    wait_key(10000)?;

    Ok(())
}

/// 开运算(先腐蚀, 在膨胀) 去除大图形外的小图形,  如果噪点大, 那么卷积核也要变大
///
/// 闭运算(先膨胀, 后腐蚀) 去除大图形内的小图形
fn kaiyunsuan() -> opencv::Result<()> {
    let m = get_structuring_element(MORPH_RECT, Size::new(7, 7), Point::new(-1, -1))?;

    let raw_mat = imread("C:\\Users\\Administrator\\Desktop\\dotj.png", IMREAD_COLOR)?;

    let mut new_mat = Mat::default();
    morphology_ex(
        &raw_mat,
        &mut new_mat,
        MORPH_RECT,
        &m,
        Point::new(-1, -1),
        1,
        BORDER_CONSTANT,
        Scalar::from(0),
    )?;

    imshow("asdsa3", &new_mat)?;

    wait_key(10000)?;

    imshow("asdsa23", &raw_mat)?;

    wait_key(10000)?;

    Ok(())
}

/// 梯度计算(原图 - 腐蚀)   求边缘
///
/// 边缘的清晰与否和卷积核大小有关, 卷积核越小, 边缘越清晰
fn tidujisuan() -> opencv::Result<()> {
    let m = get_structuring_element(MORPH_RECT, Size::new(5, 5), Point::new(-1, -1))?;

    let raw_mat = imread("C:\\Users\\Administrator\\Desktop\\j.png", IMREAD_COLOR)?;

    let mut new_mat = Mat::default();
    morphology_ex(
        &raw_mat,
        &mut new_mat,
        MORPH_GRADIENT,
        &m,
        Point::new(-1, -1),
        1,
        BORDER_CONSTANT,
        Scalar::from(0),
    )?;

    imshow("asdsa3", &new_mat)?;

    wait_key(10000)?;

    imshow("asdsa23", &raw_mat)?;

    wait_key(10000)?;

    Ok(())
}

/// 顶帽计算(原图 - 开运算)  获取大图形外的小图形
///
/// 黑帽计算(原图 - 开运算)  获取大图形内的小图形
fn dingmaojisuan() -> opencv::Result<()> {
    let m = get_structuring_element(MORPH_RECT, Size::new(7, 7), Point::new(-1, -1))?;

    let raw_mat = imread(
        "C:\\Users\\Administrator\\Desktop\\dotinj.png",
        IMREAD_COLOR,
    )?;

    let mut new_mat = Mat::default();
    // 顶帽计算, 卷积核必须要大, 在这张图里  tophat.png
    // morphology_ex(
    //     &raw_mat,
    //     &mut new_mat,
    //     MORPH_TOPHAT,
    //     &m,
    //     Point::new(-1, -1),
    //     1,
    //     BORDER_CONSTANT,
    //     Scalar::from(0),
    // )?;

    // 黑帽计算    dotinj.png
    morphology_ex(
        &raw_mat,
        &mut new_mat,
        MORPH_BLACKHAT,
        &m,
        Point::new(-1, -1),
        1,
        BORDER_CONSTANT,
        Scalar::from(0),
    )?;

    imshow("asdsa3", &new_mat)?;

    wait_key(10000)?;

    imshow("asdsa23", &raw_mat)?;

    wait_key(10000)?;

    Ok(())
}

fn main() -> opencv::Result<()> {
    // erzhihua()?;
    // zishiying()?;
    // fushi()?;
    // juanjihe()?;
    // kaiyunsuan()?;
    // tidujisuan()?;
    dingmaojisuan()?;

    Ok(())
}
