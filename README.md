# BakalarskaPraceAR
Kódy pro praktickou část mé bakalářské práce.

Jsou zde kódy pro různé aplikace rozšířené reality. Jedním je vykreslování virtuálního objektu na šachovnici, druhým na aruco kód a třetím je vizuální odometrie.

Nejprve je nutné spustit program CameraCalib, pro který je třeba vytisknout šachovnici a z několika úhlů a vzdáleností ji vyfotit kamerou, kterou chcete dále používat. Program z těchto fotek získá kalibrační parametry kamery a její matici, které pak budou dále použity.

Další program je GettingPics, ve kterém je možno uřčit po jakých časových intervalich a po jakou dobu bude ukládat snímky z kamery. Ty se budou hodit pro vizuální odometrii i pro vykreslování objektu na šachovnici.

ChessboardAsMarker je program, který si jako vstup bere fotky šachovnice umístěné v reálném prostředí a vykresluje do nich krychli s červenou a modrou horní úhlopříčkou, aby bylo možné sledovat její natočení.

CodeAsMarker nepotřebuje předem uložené snímky. Na vytištěný aruco kód položený v reálném prostředí vykresluje 2D obraz v reálném čase. Snímky s vykresleným virtuálním obrazem i ukládá.

MonoVisualOdometry provádí vizuální odometrii a jeho výstupem je trajektorie. Parametry kamery jsou zde nastaveny pro dataset ze stránky http://www.cvlibs.net/datasets/kitti/eval_odometry.php. Pokud chcete kód použít na vlastním datasetu, musíte přepsat parametry kamery, které získáte z CameraCalib.
