import sys
import re
import os

subfiles = []
Title = "Default"
Authors = []


def ParseVitalParts(vp):
    part_list = vp.split(',')
    part_list = [x.strip() for x in part_list]
    singular = [x for x in part_list if not '#' in x]
    Engines = [x for x in part_list if 'Engine' in x]
    Radiators = [x for x in part_list if 'Radiator' in x]
    OilCooler = [x for x in part_list if 'Oil Cooler' in x]
    OilPan = [x for x in part_list if 'Oil Pan' in x]
    OilTank = [x for x in part_list if 'Oil Tank' in x]
    Guns = [x for x in part_list if 'Weapon' in x]
    Pilot = [x for x in part_list if ': Pilot' in x]
    CoPilot = [x for x in part_list if 'Co-Pilot' in x]
    Gunner = [x for x in part_list if 'Gunner' in x]
    Bombadier = [x for x in part_list if 'Bombadier' in x]
    Aircrew = [x for x in part_list if 'Aircrew' in x]

    output = []

    def append(arr, str):
        if len(arr) == 1:
            output.append(str)
        elif len(arr) > 1:
            output.append('x{} {}s'.format(len(arr), str))

    append(Engines, 'Engine')
    append(OilCooler, 'Oil Cooler')
    append(OilPan, 'Oil Pan')
    append(OilTank, 'Oil Tank')
    append(Radiators, 'Radiator')
    append(Guns, 'Gun')
    output = output + singular
    str = ', '.join(output) + '\\\\'
    output = []
    append(Pilot, 'Pilot')
    append(CoPilot, 'Co-Pilot')
    append(Bombadier, 'Bombadier')
    append(Gunner, 'Gunner')
    append(Aircrew, 'Aircrew')
    return str + ', '.join(output)


def MakeSubfile(osfile):
    print("Subfile {}".format(osfile))
    with open(osfile, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    lines = [x.strip() for x in lines]
    lines = [re.sub('#', '\#', x) for x in lines]

    FileName = re.sub('[^0-9a-zA-Z]', '_', lines[1])

    Link = lines[0]
    AircraftName = lines[1]
    Nickname = lines[2]
    Cost = lines[3]
    Upkeep = lines[4]

    FirstRow, FullBoost, FullHandling, FullClimb, FullStall, FullSpeed = lines[6].split(
        '\t')
    if 'Full Fuel' in FirstRow:
        _, HalfBoost, HalfHandling, HalfClimb, HalfStall, HalfSpeed = lines[7].split(
            '\t')
        _, _, EmptyBoost, EmptyHandling, EmptyClimb, EmptyStall, EmptySpeed = lines[8].split(
            '\t')
        VitalParts = ParseVitalParts(lines[11])
        FirstLine = lines[13]
        SecondLine = lines[15]
        ThirdLine = lines[17]
        SpecialRules = '\\\\'.join(lines[19:])

        StatTable = f"""
            Full Fuel & {FullBoost} & {FullHandling} & {FullClimb} & {FullStall} & {FullSpeed} \\\\
            Half Fuel & {HalfBoost} & {HalfHandling} & {HalfClimb} & {HalfStall} & {HalfSpeed} \\\\
            Empty & {EmptyBoost} & {EmptyHandling} & {
                EmptyClimb} & {EmptyStall} & {EmptySpeed}
            \\CodeAfter
            \\begin{{tikzpicture}}
                    \\draw [thick] (2-|2) -- (2-|7) ;
                    \\draw [thick] (3-|2) -- (3-|7) ;
                    \\draw [thick] (4-|2) -- (4-|7) ;
                    \\draw [thick] (5-|2) -- (5-|7) ;
                    \\draw [thick] (2-|2) -- (last-|2) ;
                    \\draw [thick] (2-|3) -- (last-|3) ;
                    \\draw [thick] (2-|4) -- (last-|4) ;
                    \\draw [thick] (2-|5) -- (last-|5) ;
                    \\draw [thick] (2-|6) -- (last-|6) ;
                    \\draw [thick] (2-|7) -- (last-|7) ;"""
    else:
        FwBBoost, FwBHandling, FwBClimb, FwBStall, FwBSpeed = (
            FullBoost, FullHandling, FullClimb, FullStall, FullSpeed)
        _, HwBBoost, HwBHandling, HwBClimb, HwBStall, HwBSpeed = lines[7].split(
            '\t')
        _, FullBoost, FullHandling, FullClimb, FullStall, FullSpeed = lines[8].split(
            '\t')
        _, HalfBoost, HalfHandling, HalfClimb, HalfStall, HalfSpeed = lines[9].split(
            '\t')
        _, _, EmptyBoost, EmptyHandling, EmptyClimb, EmptyStall, EmptySpeed = lines[10].split(
            '\t')
        VitalParts = ParseVitalParts(lines[13])
        FirstLine = lines[15]
        SecondLine = lines[17]
        ThirdLine = lines[19]
        SpecialRules = '\\\\'.join(lines[21:])

        StatTable = f"""
            Full Load & {FwBBoost} & {FwBHandling} & {FwBClimb} & {FwBStall} & {FwBSpeed} \\\\
            \\nicefrac{{1}}{{2}}, Bombs & {HwBBoost} & {HwBHandling} & {HwBClimb} & {HwBStall} & {HwBSpeed} \\\\
            Full Fuel & {FullBoost} & {FullHandling} & {FullClimb} & {FullStall} & {FullSpeed} \\\\
            Half Fuel & {HalfBoost} & {HalfHandling} & {HalfClimb} & {HalfStall} & {HalfSpeed} \\\\
            Empty & {EmptyBoost} & {EmptyHandling} & {
                EmptyClimb} & {EmptyStall} & {EmptySpeed}
            \\CodeAfter
            \\begin{{tikzpicture}}
                    \\draw [thick] (2-|2) -- (2-|7) ;
                    \\draw [thick] (3-|2) -- (3-|7) ;
                    \\draw [thick] (4-|2) -- (4-|7) ;
                    \\draw [thick] (5-|2) -- (5-|7) ;
                    \\draw [thick] (6-|2) -- (6-|7) ;
                    \\draw [thick] (7-|2) -- (7-|7) ;
                    \\draw [thick] (2-|2) -- (last-|2) ;
                    \\draw [thick] (2-|3) -- (last-|3) ;
                    \\draw [thick] (2-|4) -- (last-|4) ;
                    \\draw [thick] (2-|5) -- (last-|5) ;
                    \\draw [thick] (2-|6) -- (last-|6) ;
                    \\draw [thick] (2-|7) -- (last-|7) ;"""

    with open('./templates/Sub.pretex', 'r', encoding='utf-8') as file:
        str = file.read()

    str = str.format(
        FileName=FileName,
        AcftName=AircraftName,
        Cost=Cost,
        Upkeep=Upkeep,
        Description=Nickname,
        StatTable=StatTable,
        VitalParts=VitalParts,
        Stats1=FirstLine,
        Stats2=SecondLine,
        Stats3=ThirdLine,
        SpecialRules=SpecialRules,
        Link=Link
    )
    subfiles.append(FileName+'.tex')
    with open('./subfiles/'+FileName+'.tex', 'w', encoding='utf-8') as file:
        file.writelines(str)

    if not os.path.isfile('./desc/'+FileName+'_desc.txt'):
        with open('./desc/'+FileName+'_desc.txt', 'w', encoding='utf-8') as file:
            file.write(
                '\\documentclass[../subfiles/'+FileName+'.tex]{subfiles}\n')
            file.write('\\begin{document}\n')
            file.write('\n')
            file.write('\\end{document}\n')

    if not os.path.isfile('./desc/'+FileName+'_table.txt'):
        with open('./desc/'+FileName+'_table.txt', 'w', encoding='utf-8') as file:
            file.write("""Role=Edit, Add or
Served With=remove lines
First Flight=to fill
Strengths=out the
Weaknesses=table
Inspiration=like this.""")

    has_image = False
    for file in os.listdir('./images/'):
        if FileName+'_image' in file:
            has_image = True
            break
    if not has_image:
        with open('./images/'+FileName+'_image.png', 'wb') as img_out, open('./templates/Default.png', 'rb') as img_in:
            img_out.write(img_in.read())


if not os.path.isdir('./desc'):
    os.mkdir('./desc')
if not os.path.isdir('./images'):
    os.mkdir('./images')
if not os.path.isdir('./subfiles'):
    os.mkdir('./subfiles')

if os.path.isfile('./AuthorInfo.text'):
    with open('./AuthorInfo.text', 'r', encoding='utf-8') as AI:
        Title = AI.readline()
        Title = Title.strip()
        for line in AI:
            Authors.append(line)
else:
    Title = input('Title: ')
    Title = Title.strip()
    auth = 'stuff'
    while auth:
        auth = input("Author (press enter to end entry): ")
        if auth:
            Authors.append(auth)
    with open('./AuthorInfo.text', 'w', encoding='utf-8') as AI:
        AI.write(Title+'\n')
        for auth in Authors:
            AI.write(auth+'\\\\')

for file in os.listdir('.'):
    if file.endswith('.txt'):
        MakeSubfile(file)

with open('./templates/Main.pretex', 'r', encoding='utf-8') as main_in, open(Title+'.tex', 'w', encoding='utf-8') as main_out:
    str = main_in.read()
    AuthorStr = '\\'.join(Authors)
    SubfileString = ''
    for sf in subfiles:
        SubfileString += f"\\subfile{{./subfiles/{sf}}}\n"
    str = str.format(
        Title=Title,
        Authors=AuthorStr,
        SubfileIncludes=SubfileString
    )
    main_out.write(str)

os.system(
    f'lualatex -synctex=1 -interaction=nonstopmode -file-line-error -pdf "{Title}.tex"')
os.system(
    f'lualatex -synctex=1 -interaction=nonstopmode -file-line-error -pdf "{Title}.tex"')
os.system(
    f'lualatex -synctex=1 -interaction=nonstopmode -file-line-error -pdf "{Title}.tex"')
