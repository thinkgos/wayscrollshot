mod capture;
mod cli;
mod constants;
mod overlay;
mod output;
mod region_overlay;
mod session;
mod stitch;
mod types;

use anyhow::Result;

use crate::cli::Args;

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse_args();
    session::run(args)
}
