use std::{env, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=path/to/Cargo.lock");

    // get $USERPRFILE$ environment variable
    let user_home_output = Command::new("powershell")
        .args(["/C", "$Env:USERPROFILE"])
        .output()
        .unwrap();
    let user_home = std::str::from_utf8(user_home_output.stdout.as_slice())
        .unwrap()
        .trim();

    // get toolchain name for find std library directory
    let toolchain_output = Command::new("powershell")
        .args(["/C", "rustup", "default"])
        .output()
        .unwrap();
    let toolchain = std::str::from_utf8(toolchain_output.stdout.as_slice()).unwrap();
    let toolchain = toolchain.split_ascii_whitespace().next().unwrap().trim();

    // get host name
    let default_host = env::var("HOST").unwrap();

    println!("user_home : {}", user_home);
    println!("toolchain : {}", toolchain);
    println!("default_host : {}", default_host);

    let lib_dir =
        format!("{user_home}\\.rustup\\toolchains\\{toolchain}\\lib\\rustlib\\{default_host}\\lib");
    println!("lib_dir : {}", lib_dir);

    let file_name = std::fs::read_dir(lib_dir.as_str())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .filter(|file_name| file_name.starts_with("std") && file_name.ends_with(".dll"))
        .next()
        .unwrap();

    let file_path = format!("{}\\{}", lib_dir, file_name);

    let out_dir = format!(
        "{}\\target\\{}",
        env::var("CARGO_MANIFEST_DIR").unwrap(),
        env::var("PROFILE").unwrap()
    );
    println!("out_dir : {}", out_dir);

    // copy std library to target directory
    match std::fs::copy(file_path, format!("{}\\{}", out_dir, file_name)) {
        Ok(_) => println!("succeeded copy"),
        _ => (),
    };
}
