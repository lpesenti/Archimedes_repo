from pipython import GCSDevice

pidevice = GCSDevice('C-884')
pidevice.InterfaceSetupDlg()
print(pidevice.qIDN())
pidevice.CloseConnection()