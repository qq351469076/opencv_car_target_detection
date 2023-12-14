use opencv::core::{
    add, no_array, Point, Point2f, Rect, Size, Size_, BORDER_CONSTANT, BORDER_DEFAULT, CV_32FC1,
    CV_64F, CV_64FC1, CV_64FC3, CV_64FC4, CV_8UC1, DECOMP_LU,
};
use opencv::highgui::{imshow, wait_key};
use opencv::imgcodecs::{imread, IMREAD_COLOR};
use opencv::imgproc::{
    bilateral_filter, blur, canny, filter_2d, gaussian_blur, get_perspective_transform,
    get_rotation_matrix_2d, laplacian, median_blur, resize, sobel, warp_affine, warp_perspective,
    INTER_AREA, INTER_LINEAR,
};
use opencv::prelude::*;

/// 放大缩小图片
fn resize_func() -> opencv::Result<()> {
    let mat = imread("C:\\Users\\Administrator\\Desktop\\1.png", IMREAD_COLOR)?;

    let mut new = Mat::default();

    // resize(
    //     &mat,
    //     &mut new,
    //     Size {
    //         width: 1024,
    //         height: 768,
    //     },
    //     0f64,
    //     0f64,
    //     INTER_LINEAR,
    // )?;

    resize(
        &mat,
        &mut new,
        Size {
            width: 0,
            height: 0,
        },
        0.9f64,
        0.9f64,
        INTER_AREA,
    )?;
    imshow("asdsa", &new)?;
    wait_key(10000)?;

    Ok(())
}

/// 仿射变换
///
/// 图像旋转, 平移或放大缩小的过程叫仿射变换
///
/// 平移需要2*3的矩阵
fn fangshebianhuan() -> opencv::Result<()> {
    let mat = imread("C:\\Users\\Administrator\\Desktop\\dog.jpeg", IMREAD_COLOR)?;

    // 通过切片创造Mat
    // [1, 0, 100]    100是通过x轴向右偏移100个像素
    // [0, 1,  0 ]    最后的0没有含义
    // let m = Mat::from_slice_2d(&[[1f32, 0f32, 100f32], [0f32, 1f32, 0f32]])?;

    // 变换矩阵工具, 可代替上面的操作
    // 通常是逆时针旋转, 1.0代表不缩放
    let m = get_rotation_matrix_2d(Point2f::new(100f32, 100f32), 15_f64, 1_f64)?;

    let mut new_mat = Mat::default();
    warp_affine(
        &mat,
        &mut new_mat,
        &m,
        Size_ {
            width: mat.cols(),
            height: mat.rows(),
        },
        INTER_LINEAR,
        BORDER_CONSTANT,
        0.into(),
    )?;

    imshow("adasdas", &new_mat)?;

    wait_key(10000)?;

    Ok(())
}

/// 透视变换
///
/// 将一个坐标系变成另外的坐标系(小学课本拍张铺平)
fn toushibianhuan() -> opencv::Result<()> {
    let mat = imread("C:\\Users\\Administrator\\Desktop\\2.jpeg", IMREAD_COLOR)?;

    // 通过切片创造Mat
    // 四个数组分别为图中的四个角
    let src = Mat::from_slice_2d(&[
        [100f32, 1100f32],
        [2100f32, 1100f32],
        [0f32, 4000f32],
        [2500f32, 3900f32],
    ])?;

    // 要放到屏幕哪个地方, 以0,0左上角为起点
    let dst = Mat::from_slice_2d(&[
        [0f32, 0f32],
        [2300f32, 0f32],
        [0f32, 3000f32],
        [2300f32, 3000f32],
    ])?;

    let m = get_perspective_transform(&src, &dst, DECOMP_LU)?;

    let mut new_mat = Mat::default();
    warp_perspective(
        &mat,
        &mut new_mat,
        &m,
        Size_ {
            // 这里自然以透视变换后的实际长度和宽度为准
            width: 2300,
            height: 3000,
        },
        INTER_LINEAR,
        BORDER_CONSTANT,
        0.into(),
    )?;

    imshow("adasdas", &new_mat)?;

    wait_key(10000)?;

    Ok(())
}

/// 低通滤波
///
/// 降噪和平滑图像
fn juanji() -> opencv::Result<()> {
    let mat = imread("C:\\Users\\Administrator\\Desktop\\1.png", IMREAD_COLOR)?;

    let calc = 1.0 / 25.0;

    // 5 x 5 内核
    let kernel = Mat::from_slice_2d(&[
        [calc, calc, calc, calc, calc],
        [calc, calc, calc, calc, calc],
        [calc, calc, calc, calc, calc],
        [calc, calc, calc, calc, calc],
        [calc, calc, calc, calc, calc],
    ])?;

    let mut new_mat = Mat::default();

    filter_2d(
        &mat,
        &mut new_mat,
        -1,
        &kernel,
        Point::new(-1, -1),
        0.0,
        BORDER_DEFAULT,
    )?;

    imshow("adssa", &new_mat)?;

    wait_key(10000)?;

    Ok(())
}

/// 低通 - 均值滤波
fn junzhilvbo() -> opencv::Result<()> {
    let mut new_mat = Mat::default();

    let mat = imread("C:\\Users\\Administrator\\Desktop\\1.png", IMREAD_COLOR)?;

    blur(
        &mat,
        &mut new_mat,
        Size::new(5, 5),
        Point::new(-1, -1),
        BORDER_DEFAULT,
    )?;

    imshow("asdsa", &new_mat)?;

    wait_key(10000)?;

    Ok(())
}

/// 低通 - 高斯滤波(解决高斯噪音, 小噪音, 不是胡椒那么大的)
fn gaosilvbo() -> opencv::Result<()> {
    let mut new_mat = Mat::default();

    let mat = imread("C:\\Users\\Administrator\\Desktop\\3.png", IMREAD_COLOR)?;

    gaussian_blur(
        &mat,
        &mut new_mat,
        Size::new(5, 5),
        1f64,
        0f64,
        BORDER_DEFAULT,
    )?;

    imshow("asdsa", &mat)?;
    wait_key(10000)?;

    imshow("asdsa", &new_mat)?;

    wait_key(10000)?;

    Ok(())
}

/// 低通 - 中值滤波(对胡椒噪音效果明显)
///
///取中间值作为卷积后的结果值
fn zhongzhilvbo() -> opencv::Result<()> {
    let mut new_mat = Mat::default();

    let mat = imread("C:\\Users\\Administrator\\Desktop\\3.png", IMREAD_COLOR)?;

    median_blur(&mat, &mut new_mat, 5)?;

    imshow("asdsa", &new_mat)?;

    wait_key(10000)?;

    Ok(())
}

/// 低通 - 双边滤波(美颜, 保留边缘)
fn shuangbianlvbo() -> opencv::Result<()> {
    let mut new_mat = Mat::default();

    let mat = imread("C:\\Users\\Administrator\\Desktop\\lena.png", IMREAD_COLOR)?;

    bilateral_filter(&mat, &mut new_mat, 7, 20f64, 50f64, BORDER_DEFAULT)?;

    imshow("asdsa3", &mat)?;

    wait_key(10000)?;

    imshow("asdsa", &new_mat)?;

    wait_key(10000)?;

    Ok(())
}

/// 高通 - 索贝尔(x, y都要求一遍, 然后再相加)
///
/// kisize设置成-1会变成Scharr算法, 能识别更细小的线
fn suobeier() -> opencv::Result<()> {
    let mut x = Mat::default();
    let mut y = Mat::default();

    let raw_mat = imread("C:\\Users\\Administrator\\Desktop\\chess.png", IMREAD_COLOR)?;

    sobel(&raw_mat, &mut x, -1, 0, 1, 5, 1f64, 0f64, BORDER_DEFAULT)?;
    sobel(&raw_mat, &mut y, -1, 1, 0, 5, 1f64, 0f64, BORDER_DEFAULT)?;

    let mut new_mat = Mat::default();
    add(&x, &y, &mut new_mat, &no_array(), -1)?;

    imshow("asdsa3", &new_mat)?;

    wait_key(10000)?;

    Ok(())
}

/// 高通 - 拉普拉斯算子
///
/// 可同时对x, y进行推导, 缺点是对噪音敏感, 一般需要先进行去噪之后再使用拉普拉斯
fn lapulasi() -> opencv::Result<()> {
    let mut new_mat = Mat::default();

    let raw_mat = imread("C:\\Users\\Administrator\\Desktop\\chess.png", IMREAD_COLOR)?;

    laplacian(&raw_mat, &mut new_mat, -1, 1, 1f64, 0f64, BORDER_DEFAULT)?;

    imshow("asdsa3", &new_mat)?;

    wait_key(10000)?;

    Ok(())
}

/// Canny边缘检测
///
/// 使用5 x 5高斯滤波消除噪声
///
/// 计算图像梯度方向(0°,45°, 90°, 135°)
///
///取局部最大值
///
/// 阈值计算
fn canny_bianyuanjiance() -> opencv::Result<()> {
    let mut new_mat = Mat::default();

    let raw_mat = imread("C:\\Users\\Administrator\\Desktop\\chess.png", IMREAD_COLOR)?;

    canny(&raw_mat, &mut new_mat, 200f64, 400f64, 3, false)?;

    imshow("asdsa3", &new_mat)?;

    wait_key(10000)?;

    Ok(())
}

fn main() -> opencv::Result<()> {
    // resize_func()?;
    // fangshebianhuan()?;
    // toushibianhuan()?;
    // juanji()?;
    // junzhilvbo()?;
    // gaosilvbo()?;
    // zhongzhilvbo()?;
    // shuangbianlvbo()?;
    // suobeier()?;
    // lapulasi()?;
    canny_bianyuanjiance()?;

    Ok(())
}
