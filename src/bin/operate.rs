use opencv::core::{
    add, bitwise_and, bitwise_not, bitwise_or, bitwise_xor, no_array, Rect, Scalar, CV_8UC1,
    CV_8UC3,
};
use opencv::highgui::{imshow, wait_key};
use opencv::imgcodecs::{imread, IMREAD_COLOR};
use opencv::prelude::*;

fn roi_3_channel() -> opencv::Result<()> {
    // ROI 赋值
    let mat = Mat::new_rows_cols_with_default(200, 200, CV_8UC1, 0.into())?;
    Mat::roi(&mat, Rect::new(50, 50, 100, 100))?.set_scalar((0, 0, 0).into())?;
    Ok(())
}

fn roi_1_channel() -> opencv::Result<()> {
    // ROI 赋值3通道
    let mat = Mat::new_rows_cols_with_default(200, 200, CV_8UC1, 0.into())?;
    Mat::roi(&mat, Rect::new(50, 50, 100, 100))?.set_scalar(Scalar::from(255))?;
    Ok(())
}

// 非运算(取反)
fn wise_not() -> opencv::Result<()> {
    // ROI 赋值3通道
    let mat = Mat::new_rows_cols_with_default(200, 200, CV_8UC1, 0.into())?;
    Mat::roi(&mat, Rect::new(50, 50, 100, 100))?.set_scalar(Scalar::from(255))?;

    let mut ddd = Mat::default();

    bitwise_not(&mat, &mut ddd, &no_array()).unwrap();

    imshow("adasd", &ddd).unwrap();

    wait_key(10000).unwrap();

    Ok(())
}

// 与运算(取交集)(同时为1, 则为真)
fn wise_and() -> opencv::Result<()> {
    // ROI 赋值3通道
    let one = Mat::new_rows_cols_with_default(200, 200, CV_8UC1, 0.into())?;
    Mat::roi(&one, Rect::new(20, 20, 120, 120))?.set_scalar(Scalar::from(255))?;

    let two = Mat::new_rows_cols_with_default(200, 200, CV_8UC1, 0.into())?;
    Mat::roi(&two, Rect::new(60, 60, 120, 120))?.set_scalar(Scalar::from(255))?;

    let mut final_mat = Mat::default();

    bitwise_and(&one, &two, &mut final_mat, &no_array()).unwrap();

    imshow("adasd", &final_mat).unwrap();

    wait_key(10000).unwrap();

    Ok(())
}

// 或运算(取两个集合的所有值)
fn wise_or() -> opencv::Result<()> {
    // ROI 赋值3通道
    let one = Mat::new_rows_cols_with_default(200, 200, CV_8UC1, 0.into())?;
    Mat::roi(&one, Rect::new(20, 20, 120, 120))?.set_scalar(Scalar::from(255))?;

    let two = Mat::new_rows_cols_with_default(200, 200, CV_8UC1, 0.into())?;
    Mat::roi(&two, Rect::new(60, 60, 120, 120))?.set_scalar(Scalar::from(255))?;

    let mut final_mat = Mat::default();

    bitwise_or(&one, &two, &mut final_mat, &no_array()).unwrap();

    imshow("adasd", &final_mat).unwrap();

    wait_key(10000).unwrap();

    Ok(())
}

// 异或运算(交集部分为0 , 不交集地方为1)
fn wise_xor() -> opencv::Result<()> {
    // ROI 赋值3通道
    let one = Mat::new_rows_cols_with_default(200, 200, CV_8UC1, 0.into())?;
    Mat::roi(&one, Rect::new(20, 20, 120, 120))?.set_scalar(Scalar::from(255))?;

    let two = Mat::new_rows_cols_with_default(200, 200, CV_8UC1, 0.into())?;
    Mat::roi(&two, Rect::new(60, 60, 120, 120))?.set_scalar(Scalar::from(255))?;

    let mut final_mat = Mat::default();

    bitwise_xor(&one, &two, &mut final_mat, &no_array()).unwrap();

    imshow("adasd", &final_mat).unwrap();

    wait_key(10000).unwrap();

    Ok(())
}

fn add_logo() -> opencv::Result<()> {
    let logo = Mat::new_rows_cols_with_default(200, 200, CV_8UC3, 0.into())?;
    Mat::roi(&logo, Rect::new(20, 20, 120, 120))?.set_scalar((0, 0, 255).into())?;
    Mat::roi(&logo, Rect::new(60, 60, 120, 120))?.set_scalar((0, 255, 0).into())?;

    let mask = Mat::new_rows_cols_with_default(200, 200, CV_8UC1, 0.into())?;
    Mat::roi(&mask, Rect::new(20, 20, 120, 120))?.set_scalar(255.into())?;
    Mat::roi(&mask, Rect::new(60, 60, 120, 120))?.set_scalar(255.into())?;

    // 取反, 目的把黑底暴漏出来(有时候黑底的图片太复杂, 于是先把简单的翻转色先弄出来)
    let mut new_mask = Mat::default();
    // 翻转颜色, 这样黑底就暴漏出来了
    bitwise_not(&mask, &mut new_mask, &no_array())?;

    let mut dog_mat = imread("C:\\Users\\Administrator\\Desktop\\dog.jpeg", IMREAD_COLOR)?;
    let roi = Mat::roi(&dog_mat, Rect::new(0, 0, 200, 200))?;

    let mut tmp = Mat::default();
    // 黑底白边和原图区域进行交集, 相当于抠图
    bitwise_and(&roi, &roi, &mut tmp, &new_mask)?;

    let mut new_logo = Mat::default();
    add(&tmp, &logo, &mut new_logo, &no_array(), -1)?;

    Mat::roi(&dog_mat, Rect::new(0, 0, 200, 200))?.set(&new_logo)?;

    imshow("asdasd", &dog_mat).unwrap();

    wait_key(10000)?;

    Ok(())
}

fn main() -> opencv::Result<()> {
    add_logo()?;

    Ok(())
}
