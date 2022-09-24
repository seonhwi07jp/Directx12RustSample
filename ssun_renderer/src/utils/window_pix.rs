use std::cell::RefCell;

use windows::{
    core::PCSTR,
    Win32::{
        Foundation::{HINSTANCE},
        System::LibraryLoader::{FreeLibrary, GetModuleHandleA, LoadLibraryA},
    },
};
/* -------------------------------------------------------------------------- */
/*                              struct WindowPIX                              */
/* -------------------------------------------------------------------------- */
pub struct WindowPIX {
    module: RefCell<Option<HINSTANCE>>,
}

impl Drop for WindowPIX {
    fn drop(&mut self) { self.disable(); }
}

/* --------------------------------- Method --------------------------------- */
impl WindowPIX {
    pub fn new() -> WindowPIX {
        WindowPIX {
            module: RefCell::new(None),
        }
    }

    pub fn enable(&self) {
        if let Ok(module) = unsafe { GetModuleHandleA(PCSTR(b"WinPixGpuCapturer.dll\0".as_ptr())) } {
            *self.module.borrow_mut() = Some(module);
        }

        let pix_path = "C:\\Program Files\\Microsoft PIX".to_string();

        let mut directories = std::fs::read_dir(&pix_path).unwrap();
        let mut newest_version: String = String::new();

        directories.all(|entry| {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_dir() {
                if newest_version.is_empty() || newest_version < entry.file_name().to_str().unwrap().to_string() {
                    newest_version = entry.path().file_name().unwrap().to_str().unwrap().to_string();
                }
            }

            true
        });

        let dll_path = pix_path + "\\" + &newest_version + "\\WinPixGpuCapturer.dll\0";

        if let Ok(module) = unsafe { LoadLibraryA(PCSTR(dll_path.as_ptr())) } {
            *self.module.borrow_mut() = Some(module);
        } else {
            unsafe { LoadLibraryA(PCSTR(dll_path.as_ptr())) }.unwrap();
        }
    }

    pub fn disable(&self) {
        let module = *self.module.borrow();

        match module {
            Some(module) => unsafe {
                FreeLibrary(module);
            },
            None => (),
        }
    }
}
