def preview_txt(cnt_line, txt_path):
    print(f"{txt_path}:")
    with open(txt_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= cnt_line:
                break
            print("=" * 30 + f" line {i + 1} " + "=" * 30)
            print(line.strip())


def add_endoftext(infile, outfile):
    if not infile.exists():
        return
    cnt_in, cnt_out = 0, 0
    with open(infile, "r", encoding="utf-8") as fin, open(outfile, "w", encoding="utf-8") as fout:
        for line in fin:
            text = line.strip()
            cnt_in += 1
            if text:
                fout.write(text + "<|endoftext|>\n")
                cnt_out += 1
    print(f"add_endoftext Complete! {cnt_out} / {cnt_in}")
