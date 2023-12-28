use opencv::core::{Scalar, Size, Vec3d, Vec3f, Vec4w};
use opencv::prelude::*;
use opencv::types::VectorOfi32;

/// 矩阵的基本属性
fn mat_default() -> opencv::Result<()> {
    let mat = Mat::default();

    println!("矩阵元素类型为{:?}", mat.typ());
    println!("矩阵元素类型大小为{:?}", mat.depth());
    println!("矩阵元素通道数为{:?}", mat.channels());
    println!("矩阵元素大小为{:?}", mat.size()?);
    println!("矩阵元素维度为{:?}", mat.dims());

    assert_eq!(u8::opencv_type(), mat.typ());
    assert_eq!(u8::opencv_depth(), mat.depth());
    assert_eq!(u8::opencv_channels(), mat.channels());
    assert_eq!(Size::new(0, 0), mat.size()?);
    assert_eq!(0, mat.dims());
    assert!(!mat.is_allocated());
    assert!(mat.data().is_null());
    Ok(())
}

/// 创建x行x列的二维数组(unsafe), 并获取指定位置
/// 用于在给定的Mat对象上动态分配新的数组数据。它使用了一种机制来管理内存，通过释放以前的数据并为新的数据分配内存。它返回一个结果来指示分配是否成功。
fn mat_create() -> opencv::Result<()> {
    let mut mat = Mat::default();
    // 创建10行10列的u16矩阵
    unsafe { mat.create_rows_cols(10, 10, u16::opencv_type())? };
    assert!(mat.is_allocated());
    assert!(!mat.data().is_null());
    // 将所有元素设置成7
    mat.set_scalar(7.into())?;
    // 获取指定2维数组指定位置的元素
    assert_eq!(7, *mat.at_2d::<u16>(0, 0)?);
    assert_eq!(7, *mat.at_2d::<u16>(3, 3)?);
    assert_eq!(7, *mat.at_2d::<u16>(9, 9)?);
    mat.release()?;
    assert!(!mat.is_allocated());
    assert!(mat.data().is_null());
    assert_eq!(Size::new(0, 0), mat.size()?);
    Ok(())
}

/// 从VectorOfi32或vec数组中创建二维Mat, 并获取指定指定位置元素
fn mat_from_iter() -> opencv::Result<()> {
    {
        let mut vec = VectorOfi32::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);
        let mat = Mat::from_exact_iter(vec.into_iter())?;
        println!("有{:?}行", mat.rows());
        println!("有{:?}列", mat.cols());
        assert_eq!(3, mat.rows());
        assert_eq!(1, mat.cols());
        assert_eq!(i32::opencv_type(), mat.typ());
        println!("元素类型为{:?}", mat.typ());
        assert_eq!(1, *mat.at_2d::<i32>(0, 0)?);
        assert_eq!(2, *mat.at_2d::<i32>(1, 0)?);
        assert_eq!(3, *mat.at_2d::<i32>(2, 0)?);
    }

    {
        let vec: Vec<i32> = vec![1, 2, 3];
        let mat = Mat::from_exact_iter(vec.into_iter())?;
        assert_eq!(3, mat.rows());
        assert_eq!(1, mat.cols());
        assert_eq!(i32::opencv_type(), mat.typ());
        assert_eq!(1, *mat.at_2d::<i32>(0, 0)?);
        assert_eq!(2, *mat.at_2d::<i32>(1, 0)?);
        assert_eq!(3, *mat.at_2d::<i32>(2, 0)?);
    }
    Ok(())
}

/// Mat类的构造函数之一。
/// 这个函数使用了一种简化的方式来创建一个新的Mat对象，使用指定的行数、列数和数据类型来初始化新的矩阵。
/// 它返回一个Mat对象。
fn mat_for_rows_and_cols() -> opencv::Result<()> {
    let mat = unsafe { Mat::new_rows_cols(400, 300, Vec3d::opencv_type()) }?;
    assert_eq!(Vec3d::opencv_type(), mat.typ());
    assert_eq!(Vec3d::opencv_depth(), mat.depth());
    println!("{:?}", mat.channels());
    assert_eq!(Vec3d::opencv_channels(), mat.channels());
    assert!(mat.is_continuous());
    assert!(!mat.is_submatrix());
    assert_eq!(Size::new(300, 400), mat.size()?);
    assert_eq!(400, mat.rows());
    assert_eq!(300, mat.cols());
    assert_eq!(2, mat.mat_size().len());
    assert_eq!(400, mat.mat_size()[0]);
    assert_eq!(300, mat.mat_size()[1]);
    assert_eq!(2, mat.dims());
    assert_eq!(2, mat.mat_step().buf().len());
    assert_eq!(7200, mat.mat_step().buf()[0]);
    assert_eq!(24, mat.mat_step().buf()[1]);
    assert_eq!(24, mat.elem_size()?);
    assert_eq!(8, mat.elem_size1());
    assert_eq!(900, mat.step1(0)?);
    assert_eq!(3, mat.step1(1)?);
    assert_eq!(120000, mat.total());
    Ok(())
}

/// 创建指定维度x行x列的数组
fn mat_nd() -> opencv::Result<()> {
    {
        // 创建指定维度x行x列的数组, 3行, 3列 类型为Vec4w类型, 全部由0填充的矩阵
        let mut mat = Mat::new_nd_with_default(&[3, 3, 3], Vec4w::opencv_type(), 0.into())?;

        // 打印 第一维度的第1行第1列的元素
        println!("{:?}", mat.at_3d::<Vec4w>(0, 1, 1)?);

        assert_eq!(0, mat.at_3d::<Vec4w>(1, 1, 1)?[0]);
        // 修改第一维度, 第一行, 第一列的元素, 将其全部设置成5
        *mat.at_3d_mut::<Vec4w>(1, 1, 1)? = Vec4w::all(10);
        assert_eq!(10, mat.at_3d::<Vec4w>(1, 1, 1)?[0]);
        assert_eq!(0, mat.at_3d::<Vec4w>(1, 1, 2)?[2]);
        assert_eq!(3, mat.dims());
        assert_eq!([3, 3, 3], *mat.mat_size());
    }

    {
        // 矩阵的长度是5040是因为您创建的矩阵（mat）是一个多维矩阵，其维度为[2, 3, 4, 5, 6, 7]。该维度的所有元素相乘得到矩阵的长度
        let dims = VectorOfi32::from_iter(vec![2, 3, 4, 5, 6, 7]);
        let mut mat = Mat::new_nd_vec_with_default(&dims, Vec4w::opencv_type(), 0.into())?;

        assert_eq!(-1, mat.rows());
        assert_eq!(-1, mat.cols());

        assert_eq!(Vec4w::default(), *mat.at_nd::<Vec4w>(&[1, 2, 3, 4, 5, 6])?);
        // 索引表示矩阵中某个元素的位置，其中第1行，第2列，第3深度，第4高度，第5宽度和第6通道。
        *mat.at_nd_mut::<Vec4w>(&[1, 2, 3, 4, 5, 6])? = Vec4w::from([5, 6, 7, 8]);
        assert_eq!(
            Vec4w::from([5, 6, 7, 8]),
            *mat.at_nd::<Vec4w>(&[1, 2, 3, 4, 5, 6])?
        );
    }

    Ok(())
}

/// 从数组引用里创建二维数组, 并且改变矩阵的形状
fn mat_at_1d() -> opencv::Result<()> {
    let s: Vec<Vec<f32>> = vec![vec![1., 2., 3.], vec![4., 5., 6.], vec![7., 8., 9.]];

    {
        // 从数组切片中创建二维数组,  3行3列
        let mat = Mat::from_slice_2d(&s)?;

        // 1个通道, 拆分成1行, 相当于1行9列
        let mut mat = mat.reshape(1, 1)?;

        assert_eq!(1, mat.rows());
        assert_eq!(9, mat.cols());

        assert_eq!(*mat.at::<f32>(0)?, 1.);
        assert_eq!(*mat.at::<f32>(5)?, 6.);
        assert_eq!(*mat.at::<f32>(8)?, 9.);
        *mat.at_mut::<f32>(4)? = 2.;
        assert_eq!(*mat.at::<f32>(4)?, 2.);
    }

    {
        let mat = Mat::from_slice_2d(&s)?;
        // 1个通道, 拆分成9行, 相当于9行1列
        let mut mat = mat.reshape(1, 9)?;
        assert_eq!(9, mat.rows());
        assert_eq!(1, mat.cols());

        assert_eq!(*mat.at::<f32>(0)?, 1.);
        assert_eq!(*mat.at::<f32>(4)?, 5.);
        assert_eq!(*mat.at::<f32>(8)?, 9.);
        *mat.at_mut::<f32>(4)? = 2.;
        assert_eq!(*mat.at::<f32>(4)?, 2.);
    }

    {
        let mut mat = Mat::from_slice_2d(&s)?;

        assert_eq!(*mat.at::<f32>(0)?, 1.);
        assert_eq!(*mat.at::<f32>(6)?, 7.);
        assert_eq!(*mat.at::<f32>(8)?, 9.);
        *mat.at_mut::<f32>(4)? = 2.;
        assert_eq!(*mat.at::<f32>(4)?, 2.);
    }
    Ok(())
}

/// 通过Size结构体创建二维数组
fn mat_2d_i0_is_rows_i1_is_cols() -> opencv::Result<()> {
    // Just a sanity check about which Mat dimension corresponds to which in Size
    let mat = Mat::new_size_with_default(Size::new(6, 5), f32::opencv_type(), 1.23.into())?;
    let size = mat.size()?;
    assert_eq!(size.width, 6);
    assert_eq!(size.height, 5);
    Ok(())
}

/// 指定行和列去创建二维数组
fn mat_at_2d() -> opencv::Result<()> {
    let mut mat = Mat::new_rows_cols_with_default(100, 100, f32::opencv_type(), 1.23.into())?;
    assert_eq!(*mat.at_2d::<f32>(0, 0)?, 1.23);
    *mat.at_2d_mut::<f32>(0, 0)? = 1.;
    assert_eq!(*mat.at_2d::<f32>(0, 0)?, 1.);

    Ok(())
}

/// 修改二维数组中某个元素的值
fn mat_at_2d_multichannel() -> opencv::Result<()> {
    let mut mat =
        Mat::new_rows_cols_with_default(100, 100, Vec3f::opencv_type(), Scalar::all(1.23))?;
    let pix = *mat.at_2d::<Vec3f>(0, 0)?;
    assert_eq!(pix[0], 1.23);
    assert_eq!(pix[1], 1.23);
    assert_eq!(pix[2], 1.23);

    *mat.at_2d_mut::<Vec3f>(0, 0)? = Vec3f::from([1.1, 2.2, 3.3]);

    let pix = *mat.at_2d::<Vec3f>(0, 0)?;
    assert_eq!(pix[0], 1.1);
    assert_eq!(pix[1], 2.2);
    assert_eq!(pix[2], 3.3);

    Ok(())
}

fn main() -> opencv::Result<()> {
    // // 矩阵的基本属性
    // mat_default()?;

    // // 创建x行x列的二维数组(unsafe), 并获取指定位置
    // mat_create()?;

    // // 从VectorOfi32或vec数组中创建二维Mat, 并获取指定指定位置元素
    // mat_from_iter()?;

    // // C++原生的创建Mat方法, 执行x行x列创建Mat
    // mat_for_rows_and_cols()?;

    // // 创建多维x行x列的数组
    // mat_nd()?;

    // // // 从数组引用里创建Mat, 并且改变矩阵的形状
    // mat_at_1d()?;

    // 通过Size结构体创建二维数组
    // mat_2d_i0_is_rows_i1_is_cols()?;

    // // 指定行和列去创建二维数组
    // mat_at_2d()?;

    mat_at_2d_multichannel()?;

    Ok(())
}
