#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loc_anh.py - Công cụ cho task 1 (dữ liệu ảnh trái cây), chạy được cả trên bộ
Mendeley lẫn ảnh tự chụp.

Cài đặt (1 lần):   pip install opencv-python numpy

4 lệnh:
  1) thongke    Đếm ảnh theo từng thư mục + kích thước ảnh (để ghi vào bảng nguồn dữ liệu)
        python loc_anh.py thongke  data/sultana

  2) loc        Chấm điểm từng ảnh (mờ / tối / cháy sáng / quá nhỏ / trùng) và xuất báo cáo
        python loc_anh.py loc  data/sultana  --ra kq_sultana  --copy

  3) mau        Rút ~50 ảnh trải đều từ rõ -> mờ để 2 người xem và chọn ngưỡng mờ
        python loc_anh.py mau  kq_sultana/diem.csv  --so 50  --ra kq_sultana/mau

  4) chotnguong Đọc nhãn của 2 người, gợi ý ngưỡng mờ
        python loc_anh.py chotnguong  kq_sultana/mau/mau_dapan.csv  nhan_A.csv  nhan_B.csv

  5) chuanhoa   Đưa mọi ảnh về CÙNG kích thước vuông + cùng chất lượng JPEG (xóa dấu vết khác máy
                chụp / khác độ phân giải; nhẹ hơn nhiều khi chia sẻ). Chỉ lấy ảnh đạt nếu có --diem
        python loc_anh.py chuanhoa  data_nhom/sultana  --ra sultana_256  --diem kq_sultana/diem.csv

Lưu ý quan trọng:
  * Điểm mờ = phương sai Laplacian. Ảnh lớn được thu nhỏ về cạnh dài tối đa 512 px trước khi
    đo (ảnh nhỏ giữ nguyên, không phóng to). Điểm mờ chỉ so sánh được TRONG CÙNG MỘT BỘ ẢNH,
    nên mỗi bộ chốt ngưỡng --mo riêng. Ngưỡng mặc định (100) CHỈ để chạy thử, phải chốt lại
    bằng lệnh `mau` + `chotnguong`.
  * Phát hiện trùng bằng dHash chỉ bắt ảnh gần như y hệt. Ảnh XOAY / LẬT (ảnh tăng cường)
    KHÔNG bị bắt, nên khi chia tập phải tách theo ảnh gốc / mã quả, không chỉ dựa vào lọc trùng.
"""
import argparse
import csv
import os
import random
import shutil
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np

DUOI_ANH = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CANH_CHUAN = 512
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# ----------------------------------------------------------------- tiện ích
def liet_ke_anh(goc):
    ds = []
    for thu_muc, _, tep in os.walk(goc):
        for t in tep:
            if os.path.splitext(t)[1].lower() in DUOI_ANH:
                ds.append(os.path.join(thu_muc, t))
    return sorted(ds)


def doc_anh(duong_dan):
    """Đọc ảnh an toàn với đường dẫn có dấu tiếng Việt trên Windows."""
    try:
        buf = np.fromfile(duong_dan, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def dhash(xam):
    nho = cv2.resize(xam, (9, 8), interpolation=cv2.INTER_AREA)
    bit = (nho[:, 1:] > nho[:, :-1]).flatten()
    return np.packbits(bit)  # 8 byte


def tien_do(i, n, moi=500):
    if i % moi == 0 or i == n:
        print(f"  ... {i}/{n}", file=sys.stderr)


# ----------------------------------------------------------------- 1) thongke
def lenh_thongke(a):
    ds = liet_ke_anh(a.thu_muc)
    if not ds:
        sys.exit("Không thấy ảnh nào trong " + a.thu_muc)
    theo_thu_muc = defaultdict(list)
    for p in ds:
        theo_thu_muc[os.path.relpath(os.path.dirname(p), a.thu_muc)].append(p)

    hang = []
    for tm, dsa in sorted(theo_thu_muc.items()):
        mau = random.Random(0).sample(dsa, min(len(dsa), a.mau))
        cd, cr = [], []
        for p in mau:
            im = doc_anh(p)
            if im is not None:
                cao, rong = im.shape[:2]
                cd.append(cao)
                cr.append(rong)
        hang.append([tm, len(dsa),
                     f"{int(np.median(cr))}x{int(np.median(cd))}" if cd else "?",
                     f"{min(min(cd), min(cr))}" if cd else "?"])
    print(f"{'thu_muc':45s} {'so_anh':>7s} {'kich_thuoc_giua':>16s} {'canh_nho_nhat':>14s}")
    for h in hang:
        print(f"{h[0]:45s} {h[1]:7d} {h[2]:>16s} {h[3]:>14s}")
    print(f"TỔNG: {len(ds)} ảnh trong {len(hang)} thư mục")
    ten_csv = "thongke_" + (os.path.basename(os.path.normpath(a.thu_muc)) or "du_lieu") + ".csv"
    with open(ten_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["thu_muc", "so_anh", "kich_thuoc_giua", "canh_nho_nhat_mau"])
        w.writerows(hang)
    print(f"Đã ghi {ten_csv} (dán vào bảng nguồn dữ liệu của nhóm).")


# ----------------------------------------------------------------- 2) loc
def lenh_loc(a):
    ds = liet_ke_anh(a.thu_muc)
    if not ds:
        sys.exit("Không thấy ảnh nào trong " + a.thu_muc)
    os.makedirs(a.ra, exist_ok=True)
    print(f"Chấm điểm {len(ds)} ảnh ...", file=sys.stderr)

    dong = []
    for i, p in enumerate(ds, 1):
        tien_do(i, len(ds))
        im = doc_anh(p)
        thumuc = os.path.relpath(os.path.dirname(p), a.thu_muc)
        if im is None:
            dong.append(dict(p=p, tm=thumuc, rong=0, cao=0, mo=0.0, sang=0.0,
                             chay=0.0, h=None, ly_do=["loi_doc"]))
            continue
        cao, rong = im.shape[:2]
        xam0 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # Ảnh lớn thu nhỏ về cạnh dài tối đa --canh; ảnh nhỏ giữ nguyên (KHÔNG phóng to,
        # vì phóng to làm ảnh trông mờ giả).
        he_so = a.canh / max(cao, rong)
        if he_so < 1:
            kt = (max(1, int(rong * he_so)), max(1, int(cao * he_so)))
            xam = cv2.resize(xam0, kt, interpolation=cv2.INTER_AREA)
        else:
            xam = xam0
        mo = float(cv2.Laplacian(xam, cv2.CV_64F).var())
        sang = float(xam.mean())
        chay = float((xam >= 250).mean())
        ly_do = []
        if min(cao, rong) < a.nho:
            ly_do.append("nho")
        if sang < a.toi:
            ly_do.append("toi")
        if sang > a.sang or chay > a.chay_sang:
            ly_do.append("chay_sang")
        # Điểm mờ không đáng tin trên ảnh quá nhỏ / quá tối / cháy sáng -> chỉ xét khi các lỗi kia không có
        if mo < a.mo and not ly_do:
            ly_do.append("mo")
        dong.append(dict(p=p, tm=thumuc, rong=rong, cao=cao, mo=mo, sang=sang,
                         chay=chay, h=dhash(xam), ly_do=ly_do))

    # Trùng lặp: so trong cùng thư mục, giữ ảnh nét hơn
    nhom = defaultdict(list)
    for d in dong:
        if d["h"] is not None:
            nhom[d["tm"]].append(d)
    for tm, dsd in nhom.items():
        dsd.sort(key=lambda d: -d["mo"])
        giu = []
        for d in dsd:
            if giu:
                kc = POP[np.bitwise_xor(np.array([g["h"] for g in giu]), d["h"])].sum(axis=1)
                if kc.min() <= a.trung:
                    d["ly_do"].append("trung")
                    continue
            giu.append(d)

    # Xuất điểm
    with open(os.path.join(a.ra, "diem.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["duong_dan", "thu_muc", "rong", "cao", "mo", "do_sang",
                    "ty_le_chay_sang", "ly_do", "dat"])
        for d in dong:
            w.writerow([d["p"], d["tm"], d["rong"], d["cao"], f"{d['mo']:.2f}",
                        f"{d['sang']:.1f}", f"{d['chay']:.3f}", ";".join(d["ly_do"]),
                        0 if d["ly_do"] else 1])

    # Báo cáo
    tong = len(dong)
    dat = sum(1 for d in dong if not d["ly_do"])
    dem = Counter(l for d in dong for l in d["ly_do"])
    theo_tm = defaultdict(lambda: [0, 0])
    for d in dong:
        theo_tm[d["tm"]][0] += 1
        theo_tm[d["tm"]][1] += 0 if d["ly_do"] else 1
    dong_bc = [
        f"Thư mục nguồn : {a.thu_muc}",
        f"Ngưỡng dùng    : mờ<{a.mo}  tối<{a.toi}  sáng>{a.sang} hoặc cháy>{a.chay_sang:.0%}"
        f"  cạnh nhỏ<{a.nho}px  trùng(Hamming<={a.trung})",
        f"Tổng ảnh       : {tong}",
        f"Ảnh đạt        : {dat} ({dat / tong:.1%})",
        f"Ảnh bị loại    : {tong - dat} ({(tong - dat) / tong:.1%})",
        "Một ảnh có thể dính nhiều lý do. Số ảnh theo lý do:",
    ]
    for k in ["mo", "toi", "chay_sang", "nho", "trung", "loi_doc"]:
        if dem.get(k):
            dong_bc.append(f"   - {k:10s}: {dem[k]} ({dem[k] / tong:.1%})")
    dong_bc.append("Theo thư mục (đạt / tổng):")
    for tm, (t, dd) in sorted(theo_tm.items()):
        dong_bc.append(f"   - {tm:40s} {dd}/{t} ({dd / t:.1%})")
    bc = "\n".join(dong_bc)
    print(bc)
    with open(os.path.join(a.ra, "bao_cao.txt"), "w", encoding="utf-8") as f:
        f.write(bc + "\n")

    if a.copy:
        for d in dong:
            nhan = "sach" if not d["ly_do"] else os.path.join("loai", d["ly_do"][0])
            dich = os.path.join(a.ra, nhan, d["tm"])
            os.makedirs(dich, exist_ok=True)
            shutil.copy2(d["p"], os.path.join(dich, os.path.basename(d["p"])))
        print(f"Đã chép ảnh vào {a.ra}/sach và {a.ra}/loai/<lý do>/ (giữ nguyên cấu trúc thư mục).")


# ----------------------------------------------------------------- 3) mau
def lenh_mau(a):
    with open(a.diem_csv, encoding="utf-8-sig") as f:
        hang = list(csv.DictReader(f))
    if not a.tat_ca:
        # chỉ xét chiều "mờ": bỏ ảnh đã dính lỗi tối / cháy sáng / nhỏ / lỗi đọc
        hang = [h for h in hang if not (set(h["ly_do"].split(";")) & {"toi", "chay_sang", "nho", "loi_doc"})]
    hang.sort(key=lambda h: float(h["mo"]))
    if len(hang) < a.so:
        a.so = len(hang)
    idx = np.linspace(0, len(hang) - 1, a.so).round().astype(int)
    chon = [hang[i] for i in idx]
    random.Random(42).shuffle(chon)  # xáo trộn để người gắn nhãn không đoán theo thứ tự

    os.makedirs(a.ra, exist_ok=True)
    with open(os.path.join(a.ra, "mau_dapan.csv"), "w", newline="", encoding="utf-8-sig") as fd:
        wd = csv.writer(fd)
        wd.writerow(["stt", "mo", "ten_file_goc"])
        for stt, h in enumerate(chon, 1):
            duoi = os.path.splitext(h["duong_dan"])[1].lower()
            shutil.copy2(h["duong_dan"], os.path.join(a.ra, f"{stt:03d}{duoi}"))
            wd.writerow([f"{stt:03d}", h["mo"], h["duong_dan"]])
    for ten in ("nhan_A.csv", "nhan_B.csv"):
        with open(os.path.join(a.ra, ten), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["stt", "du_nhan_dien"])
            for stt in range(1, a.so + 1):
                w.writerow([f"{stt:03d}", ""])
    print(f"Đã rút {a.so} ảnh vào {a.ra}.")
    print("Mỗi người mở nhan_A.csv / nhan_B.csv, xem ảnh 001...,"
          " ghi 1 nếu ảnh ĐỦ rõ để nhận ra quả và vết hỏng, 0 nếu KHÔNG. Làm độc lập, không hỏi nhau.")
    print("KHÔNG gửi mau_dapan.csv cho người gắn nhãn (có điểm mờ).")


# ----------------------------------------------------------------- 4) chotnguong
def doc_nhan(duong_dan):
    kq = {}
    with open(duong_dan, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            v = r["du_nhan_dien"].strip()
            if v in ("0", "1"):
                kq[r["stt"].strip()] = int(v)
    return kq


def lenh_chotnguong(a):
    with open(a.dapan, encoding="utf-8-sig") as f:
        mo = {r["stt"].strip(): float(r["mo"]) for r in csv.DictReader(f)}
    A, B = doc_nhan(a.nhan_a), doc_nhan(a.nhan_b)
    chung = [s for s in mo if s in A and s in B]
    if len(chung) < 10:
        sys.exit("Cần cả hai người gắn nhãn ít nhất 10 ảnh.")
    dong_y = [s for s in chung if A[s] == B[s]]
    po = len(dong_y) / len(chung)
    pa1, pb1 = np.mean([A[s] for s in chung]), np.mean([B[s] for s in chung])
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    print(f"Số ảnh cả hai đã gắn nhãn : {len(chung)}")
    print(f"Đồng ý                    : {len(dong_y)}/{len(chung)} = {po:.0%}   (Cohen's kappa = {kappa:.2f})")
    if po < 0.8:
        print("CẢNH BÁO: hai người đồng ý dưới 80%. Nên thống nhất lại tiêu chí 'đủ rõ' rồi gắn lại.")
    thua = [s for s in dong_y]
    giatri = sorted(set(mo[s] for s in thua))
    tot, ngl = -1, None
    for i, v in enumerate(giatri):
        t = v if i == 0 else (giatri[i - 1] + v) / 2
        acc = np.mean([(1 if mo[s] >= t else 0) == A[s] for s in thua])
        if acc > tot:
            tot, ngl = acc, t
    print(f"Ngưỡng mờ gợi ý (--mo)    : {ngl:.1f}   (khớp {tot:.0%} trên {len(thua)} ảnh hai người cùng nhãn)")
    print(f"Chạy lại:  python loc_anh.py loc <thu_muc> --ra <ra> --mo {ngl:.0f}")
    print("Ghi vào báo cáo: số ảnh gắn nhãn, tỉ lệ đồng ý, kappa và ngưỡng chọn.")


# ----------------------------------------------------------------- 5) chuanhoa
def lenh_chuanhoa(a):
    ds = liet_ke_anh(a.thu_muc)
    if a.diem:
        dat = set()
        with open(a.diem, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r["dat"] == "1":
                    dat.add(os.path.normcase(os.path.abspath(r["duong_dan"])))
        ds = [p for p in ds if os.path.normcase(os.path.abspath(p)) in dat]
    if not ds:
        sys.exit("Không có ảnh nào để chuẩn hóa (kiểm tra lại đường dẫn / --diem).")
    k = a.kich
    loi = 0
    for i, p in enumerate(ds, 1):
        tien_do(i, len(ds))
        im = doc_anh(p)
        if im is None:
            loi += 1
            continue
        cao, rong = im.shape[:2]
        if a.che_do == "cat":      # cắt vuông ở giữa (có thể mất rìa quả)
            c = min(cao, rong)
            y0, x0 = (cao - c) // 2, (rong - c) // 2
            out = cv2.resize(im[y0:y0 + c, x0:x0 + c], (k, k),
                             interpolation=cv2.INTER_AREA if c > k else cv2.INTER_CUBIC)
        else:                      # pad: giữ nguyên tỉ lệ, thêm viền xám (không làm méo quả)
            he = k / max(cao, rong)
            nw, nh = max(1, round(rong * he)), max(1, round(cao * he))
            nho = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA if he < 1 else cv2.INTER_CUBIC)
            out = np.full((k, k, 3), 114, np.uint8)
            y0, x0 = (k - nh) // 2, (k - nw) // 2
            out[y0:y0 + nh, x0:x0 + nw] = nho
        dich = os.path.join(a.ra, os.path.relpath(os.path.dirname(p), a.thu_muc))
        os.makedirs(dich, exist_ok=True)
        ten = os.path.splitext(os.path.basename(p))[0] + ".jpg"
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, a.chat_luong])
        buf.tofile(os.path.join(dich, ten))
    print(f"Đã chuẩn hóa {len(ds) - loi}/{len(ds)} ảnh -> {a.ra}  ({k}x{k}, JPEG q={a.chat_luong}, chế độ {a.che_do})")
    if loi:
        print(f"  {loi} ảnh không đọc được, đã bỏ qua.")


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Lọc và thống kê ảnh trái cây (task 1)")
    sub = ap.add_subparsers(dest="lenh", required=True)

    p = sub.add_parser("thongke")
    p.add_argument("thu_muc")
    p.add_argument("--mau", type=int, default=100, help="số ảnh lấy mẫu/thư mục để đo kích thước")
    p.set_defaults(f=lenh_thongke)

    p = sub.add_parser("loc")
    p.add_argument("thu_muc")
    p.add_argument("--ra", required=True)
    p.add_argument("--mo", type=float, default=100.0, help="điểm mờ tối thiểu (CHỈ để chạy thử)")
    p.add_argument("--toi", type=float, default=50.0, help="độ sáng TB (0-255) tối thiểu")
    p.add_argument("--sang", type=float, default=215.0, help="độ sáng TB tối đa")
    p.add_argument("--chay-sang", dest="chay_sang", type=float, default=0.30,
                   help="tỉ lệ điểm ảnh cháy sáng tối đa (0-1). Nền trắng sẽ cao: nâng lên nếu cần")
    p.add_argument("--nho", type=int, default=224,
                   help="cạnh ngắn tối thiểu (px). Bộ ảnh vốn nhỏ (vd Pachon) dùng --nho 0")
    p.add_argument("--canh", type=int, default=CANH_CHUAN,
                   help="ảnh lớn hơn được thu nhỏ về cạnh dài này trước khi đo mờ (mặc định 512)")
    p.add_argument("--trung", type=int, default=5, help="khoảng cách Hamming coi là trùng (0-64)")
    p.add_argument("--copy", action="store_true", help="chép ảnh vào ra/sach và ra/loai/<lý do>")
    p.set_defaults(f=lenh_loc)

    p = sub.add_parser("mau")
    p.add_argument("diem_csv")
    p.add_argument("--so", type=int, default=50)
    p.add_argument("--ra", required=True)
    p.add_argument("--tat-ca", dest="tat_ca", action="store_true",
                   help="lấy cả ảnh đã bị loại vì tối/cháy sáng/nhỏ")
    p.set_defaults(f=lenh_mau)

    p = sub.add_parser("chotnguong")
    p.add_argument("dapan")
    p.add_argument("nhan_a")
    p.add_argument("nhan_b")
    p.set_defaults(f=lenh_chotnguong)

    p = sub.add_parser("chuanhoa")
    p.add_argument("thu_muc")
    p.add_argument("--ra", required=True)
    p.add_argument("--kich", type=int, default=256, help="cạnh ảnh vuông đầu ra (mặc định 256)")
    p.add_argument("--che-do", dest="che_do", choices=["pad", "cat"], default="pad",
                   help="pad: giữ tỉ lệ + viền xám (mặc định); cat: cắt vuông ở giữa")
    p.add_argument("--chat-luong", dest="chat_luong", type=int, default=90, help="chất lượng JPEG 1-100")
    p.add_argument("--diem", help="diem.csv của lệnh loc: chỉ chuẩn hóa ảnh đã đạt")
    p.set_defaults(f=lenh_chuanhoa)

    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
