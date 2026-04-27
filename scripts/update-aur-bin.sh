#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <tag-or-version> <aur-checkout-dir>" >&2
    exit 2
fi

input_version="$1"
aur_dir="$2"
pkgver="${input_version#refs/tags/}"
pkgver="${pkgver#v}"
tag="v${pkgver}"

pkgname="wayscrollshot-bin"
project_url="https://github.com/jswysnemc/wayscrollshot"
asset="wayscrollshot-archlinux-x86_64.tar.gz"
asset_url="${project_url}/releases/download/${tag}/${asset}"
pkgdesc="A scrolling screenshot tool for Wayland"

if [[ ! -d "$aur_dir/.git" ]]; then
    echo "AUR checkout not found: $aur_dir" >&2
    exit 2
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

curl -fsSL -o "${tmpdir}/${asset}" "$asset_url"
sha256="$(sha256sum "${tmpdir}/${asset}" | awk '{print $1}')"

cat >"${aur_dir}/PKGBUILD" <<EOF
# Maintainer: Snemc-s <snemc@snemc.cn>
pkgname=${pkgname}
pkgver=${pkgver}
pkgrel=1
pkgdesc="${pkgdesc}"
arch=('x86_64')
url="${project_url}"
license=('MIT')
depends=('gcc-libs' 'glibc' 'grim' 'libxkbcommon' 'opencv' 'slurp')
optdepends=(
    'wl-clipboard: copy screenshots to the Wayland clipboard'
    'xclip: X11 clipboard fallback'
)
provides=("wayscrollshot=\${pkgver}")
conflicts=('wayscrollshot' 'wayscrollshot-git')
source=("\${url}/releases/download/v\${pkgver}/${asset}")
sha256sums=('${sha256}')

package() {
    install -Dm755 wayscrollshot "\${pkgdir}/usr/bin/wayscrollshot"
}
EOF

cat >"${aur_dir}/.SRCINFO" <<EOF
pkgbase = ${pkgname}
	pkgdesc = ${pkgdesc}
	pkgver = ${pkgver}
	pkgrel = 1
	url = ${project_url}
	arch = x86_64
	license = MIT
	depends = gcc-libs
	depends = glibc
	depends = grim
	depends = libxkbcommon
	depends = opencv
	depends = slurp
	optdepends = wl-clipboard: copy screenshots to the Wayland clipboard
	optdepends = xclip: X11 clipboard fallback
	provides = wayscrollshot=${pkgver}
	conflicts = wayscrollshot
	conflicts = wayscrollshot-git
	source = ${asset_url}
	sha256sums = ${sha256}

pkgname = ${pkgname}
EOF

echo "Updated ${pkgname} to ${pkgver}"
