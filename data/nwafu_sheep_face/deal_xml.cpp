#include <iostream>
using namespace std;

int main()
{
    int index = 1; // 5372       �ļ���
    for (int i = index; i <= 1668; i++)
    {
        char path_xml[30];
        sprintf(path_xml, "Annotations/%d.xml", i);
        freopen(path_xml, "r", stdin);
        freopen("three.txt", "a" ,stdout);     // ��������ͼƬ�±�
        string str;
        cin >> str;
        int count = 0;
        while (!str.empty())
        {
            if (str == "</annotation>")
            {
                break;
            }
            string div(str.begin() + 1, str.begin() + 5);
            if (div == "name")
            {
                count++;
            }
            if (count == 3)
            {
                cout << i << endl;
                break;
            }
            cin >> str;
        }
        // if (count < 3)
        // {
        //     freopen("one_two.txt", "a" ,stdout);   // С����������ͼƬ�±�
        //     cout << i << endl;
        // }
        
    }
    //system("pause");
    return 0;
}