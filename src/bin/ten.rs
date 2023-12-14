use opencv::core::{
    no_array, Mat, Point, Point2f, Point2i, RotatedRect, Scalar, Size, Vec4i, Vector,
    BORDER_CONSTANT, CV_8U,
};
use opencv::highgui::{imshow, wait_key};
use opencv::imgcodecs::{imread, IMREAD_COLOR};
use opencv::imgproc::{
    adaptive_threshold, approx_poly_dp, arc_length, bounding_rect, box_points, canny, contour_area,
    convex_hull_def, cvt_color, dilate, draw_contours, draw_contours_def, erode, find_contours,
    find_contours_def, find_contours_with_hierarchy_def, get_structuring_element, line,
    min_area_rect, morphology_ex, rectangle, threshold, ADAPTIVE_THRESH_GAUSSIAN_C,
    CHAIN_APPROX_SIMPLE, COLOR_BGR2GRAY, LINE_8, MORPH_BLACKHAT, MORPH_GRADIENT, MORPH_RECT,
    MORPH_TOPHAT, RETR_EXTERNAL, RETR_TREE, THRESH_BINARY, THRESH_BINARY_INV,
};
use opencv::prelude::*;
use opencv::types::{
    VectorOfBox, VectorOfPoint, VectorOfPoint2f, VectorOfPoint3f, VectorOfRange, VectorOfRect,
    VectorOfRotatedRect, VectorOfScalar, VectorOfSize, VectorOfVec4f, VectorOfVec4i,
    VectorOfVectorOfPoint, VectorOfVectorOff32, VectorOff32, VectorOff64,
};

/// 轮廓查找 + 绘制轮廓 + 面积计算 + 周长计算
///
/// 必须要二值化
fn cha_zhao_lun_kuo() -> opencv::Result<()> {
    let mut new_mat = Mat::default();

    let mut raw_mat = imread("C:\\Users\\Administrator\\Desktop\\1.png", IMREAD_COLOR)?;

    // 转成灰度图
    cvt_color(&raw_mat, &mut new_mat, COLOR_BGR2GRAY, 0)?;

    let mut new_mat_2 = Mat::default();
    // 超过100阈值, 就设置成200
    threshold(&new_mat, &mut new_mat_2, 100f64, 200f64, THRESH_BINARY)?;

    let mut contours_cv = VectorOfVectorOfPoint::new();

    // 轮廓查找
    find_contours(
        &new_mat_2,
        &mut contours_cv,
        // RETR_EXTERNAL 最外层轮廓
        RETR_TREE, // 树形轮廓
        CHAIN_APPROX_SIMPLE,
        Point::default(),
    )?;

    // 从轮廓列表中提取最外层轮廓
    let single_area = contours_cv.get(0)?;
    // 计算轮廓面积
    let result = contour_area(&single_area, false)?;

    println!("{:?}", result);
    println!("{}", contours_cv.len());

    // 计算轮廓周长
    let len = arc_length(&single_area, true)?;
    println!("{}", len);

    // 在原图上绘制轮廓
    draw_contours(
        &mut raw_mat,
        &contours_cv,
        -1,
        Scalar::from((0, 0, 255)),
        1,
        LINE_8,
        &no_array(),
        1,
        Point::default(),
    )?;

    imshow("adas", &raw_mat)?;

    wait_key(10000)?;

    Ok(())
}

/// 多边形逼近(轮廓描边, 类似一个手掌包含指缝), 用于存放特征点
///
/// 多边形凸包(轮廓描边, 类似游泳划水掌), 用于存放轮廓
fn duo_bian_xing_bin_jin() -> opencv::Result<()> {
    let mut raw_mat = imread("C:\\Users\\Administrator\\Desktop\\hand.png", IMREAD_COLOR)?;

    let mut cvt_mat = Mat::default();

    // 转成灰度图
    cvt_color(&raw_mat, &mut cvt_mat, COLOR_BGR2GRAY, 0)?;

    let mut binary_mat = Mat::default();
    // 超过100阈值, 就设置成200
    threshold(&cvt_mat, &mut binary_mat, 100f64, 200f64, THRESH_BINARY)?;

    // 轮廓查找
    let mut contours = VectorOfVectorOfPoint::new();
    let mut hierarchy = VectorOfVec4i::new();

    find_contours_with_hierarchy_def(
        &binary_mat,
        &mut contours,
        &mut hierarchy,
        // RETR_EXTERNAL 最外层轮廓
        RETR_TREE, // 树形轮廓
        CHAIN_APPROX_SIMPLE,
    )?;

    let contours = contours.get(0)?;

    // 近似结果列表
    let mut approx = VectorOfPoint::new();

    // 多边形逼近
    // approx_poly_dp(&contours, &mut approx, 20.0, true)?;

    // 多边形凸包
    convex_hull_def(&contours, &mut approx)?;

    /*
    以下将近似结果中的Point用线描绘出来
     */
    let mut count = 0;

    while count < approx.len() {
        // 最后将线描绘至起点
        if count == approx.len() - 1 {
            // 起始点是这个列表中最后一个
            let before = approx.get(count).unwrap();

            // 将终止点和列表中第一个元素进行收尾拼接
            let after = approx.get(0).unwrap();

            line(
                &mut raw_mat,
                before,
                after,
                Scalar::from((0, 0, 255)),
                1,
                LINE_8,
                0,
            )?;
        } else {
            // 起始点
            let before = approx.get(count).unwrap();

            // 终止点
            let after = approx.get(count + 1).unwrap();

            line(
                &mut raw_mat,
                before,
                after,
                Scalar::from((0, 0, 255)),
                1,
                LINE_8,
                0,
            )?;
        }

        count += 1
    }

    imshow("asdad", &raw_mat)?;

    wait_key(10000)?;

    Ok(())
}

/// 最小矩阵    可以获得角度
///
/// 最大矩阵
fn zui_xiao_zui_da_jvzhen() -> opencv::Result<()> {
    let mut raw_mat = imread(
        "C:\\Users\\Administrator\\Desktop\\hello.jpeg",
        IMREAD_COLOR,
    )?;

    let mut cvt_mat = Mat::default();

    // 转成灰度图
    // Convert to grayscale
    cvt_color(&raw_mat, &mut cvt_mat, COLOR_BGR2GRAY, 0)?;

    let mut binary_mat = Mat::default();
    // 超过100阈值, 就设置成200
    // Binarization
    threshold(&cvt_mat, &mut binary_mat, 100f64, 200f64, THRESH_BINARY)?;

    // 轮廓查找
    // Contour search
    let mut contours = VectorOfVectorOfPoint::new();
    let mut hierarchy = VectorOfVec4i::new();

    find_contours_with_hierarchy_def(
        &binary_mat,
        &mut contours,
        &mut hierarchy,
        RETR_TREE, // 树形轮廓
        CHAIN_APPROX_SIMPLE,
    )?;

    // 找里面的轮廓
    //  Find the outline inside
    let contours = contours.get(1)?;

    let r = min_area_rect(&contours)?;

    // 找到最小矩形的四个角
    // Find the four corners of the smallest rectangle
    let mut box_array = VectorOfPoint2f::with_capacity(4);
    RotatedRect::points_1(r, &mut box_array)?;

    // 将四个角的f32转换成int
    // Convert four corner Point<f32> to Point<int>
    let temp: Vector<Point2i> = box_array
        .iter()
        .map(|elem| Point2i {
            x: elem.x as i32,
            y: elem.y as i32,
        })
        .collect();

    // c源代码必须需要一层包装
    // c source code must require a layer of packaging
    let mut box_point: VectorOfVectorOfPoint = Vector::with_capacity(1);
    box_point.push(temp);

    // 绘制四个最小矩形角
    // Draw the four smallest rectangular corners
    draw_contours_def(&mut raw_mat, &box_point, 0, Scalar::from((0, 0, 255)))?;

    // 绘制最大矩形
    let max = bounding_rect(&contours)?;
    rectangle(&mut raw_mat, max, Scalar::from((0, 0, 255)), 1, LINE_8, 0)?;

    imshow("asdad", &raw_mat)?;

    wait_key(10000)?;

    Ok(())
}

fn main() -> opencv::Result<()> {
    // cha_zhao_lun_kuo()?;
    // duo_bian_xing_bin_jin()?;
    zui_xiao_zui_da_jvzhen()?;

    Ok(())
}
