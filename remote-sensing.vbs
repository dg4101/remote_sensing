For c = 1 To n
    g(b,c) = a(b) + std(b) * ((2*(c-1))/(n-1) -1)
Next c

'再配置法スタート'
kei= kei+1
'各個体とクラスターの重心との距曜の計算'
For yl = starty To endy 
    For xl = startx To endx 
        min = 10^4 '初期最低距離'
        place = XMAX*(y1-1) + x1 + head 
        For c = 1 To n 
            For b = 0 To k-1 
                Get filenum(b), place,resdata
                rdata = Asc(resdata)
                d1 = (g(b,c)- rdata)^2 
                d(c)= d (c) + d l 
            Next b 
            '距離が最低距離より小さい場合は再配置して最低距離の再設定'
            If d(c)< min Then 
                min= d(c) 
                classnum = c 
            End If 
            d(c)= 0 
        Next c 
    Next xl
Next yl
'クラスタを変えた個体数算出'
Get filenumB,place,Gcmoji
Gclassnum = Asc(Gcmoji)
If Gclassnum <> classnum Then 
    change= change + 1
    'クラスタ番号を分析結果ファイルに書き込み
    Pcmoji = Chr$(classnum) 
    Put fileenumB,place,Pcmoji 
End If 
N ext xi 
N ext yJ 
'収束条件
Pchange = 100 * (change / Z) 
If Pchenge < = calend Then 
    MsgBox "計算が終了しました"
End If 


'標準偏差が最大値のクラスタをバンドごとに求める
For b = O To k-1
    stdmax(b) = 0 
    For c = l To n 
        std(b, c)= Sqr(gg(b, c)-g(b, c)^2)
        If stdmox(b) < std(b, c) Then 
            stdmax(b) = std(b, c) 
            stdmac(b) = c 
        End If 
    Next c 
Next b  

'標準偏差が最大のクラスタを分裂し、クラスタ番号を追加して重心を再計算
For b = O To k-1
    If stdmax(b) > stdall(b) * 0.7 Then
        np = np + 1
        For fc = stdmac(b) To n + np Step n + np - stdmac(b)
            For fb = O To k-1
                g(fb,fc) = GetCenter(fb,stdmac(b))
            Next fb
        Next fc
    End If
Next b
