

    
    ncore = 1.4504
    nclad = 1.4504 - 5.5e-3

    wl0 = 1.
    k0 = 2*np.pi/wl0

    fname0 = "psfabb_sr0pt1.npy"
    u0 = np.load(fname0)
    u0 = resize(u0,(401,401))
    u0 = center(u0)
    

    u0 = normalize(u0)
    norm = overlap(u0,u0)

    plt.imshow(np.abs(u0))
    plt.show()

    core_ref = 6.

    
    #3.0050251256281406 0.5847756782878644 102.83664535057073
    #98.82871923554472

    V = LPmodes.get_V(k0,core_ref,ncore,nclad)
    modes = LPmodes.get_modes(V)

    res0 = scale_u0_lp01(u0,core_ref,ncore,nclad,wl0)
    res1 = scale_u0_all(u0,modes,core_ref,ncore,nclad,norm)

    print(res0)
    print(res1)

    ### strehl 0.7 ###
    ## @ 10 um core
    # width for lp01 opt : 155.25276479964893
    # width for all opt  : 66.68094204549682

    ## @ 6 um core
    # width for lp01 opt : 104.81607423592817
    # width for all opt  : 56.68771592222539

    ## @ 5 um core
    # width for lp01 opt : 93.30509266785047
    # width for all opt  : 84.79169552149993

    ### strehl 0.27 ###
    ## @ 10 um core
    # width for lp01 opt : 147.00475271381703
    # width for all opt  : 57.83239071626544

    ## @ 6 um core
    # width for lp01 opt : 99.07351634654545
    # width for all opt  : 57.49423667423401

    ## @ 5 um core
    # width for lp01 opt : 88.15258812342618
    # width for all opt  : 74.9626104514246

    ### strehl 0.1 ###
    ## @ 10 um core
    # width for lp01 opt : 152.67450017019092
    # width for all opt  : 54.11358022084979

    ## @ 5 um core
    # width for lp01 opt : 92.40515933268863
    # width for all opt  : 81.31072661859011

    ### strehl 1 ###
    ## @ 10 um core
    # width for lp01 opt : 234.81866214245008
    # width for all opt  : 105.21162234295359

    #w0 = res0[2]
    #w1 = 105.21162234295359

    #plt.imshow(np.abs(lp01),extent=(-w2,w2,-w2,w2))
    #plt.show()


    field1 = np.zeros_like(u0,dtype=np.complex128)

    coeffs = [  [0.20912591914845202-0.12185648843849013j],
                [0.6952673875010204-0.4276556750218078j],
                [0.16968609212409883-0.10967925911704372j],
                [0.06039553779501969-0.03824586901798927j,0.02629656884801027-0.014107658711864616j],
                [0.031953964951564884-0.03434596980466746j,0.023079754914827354-0.025310820904414093j],
                [0.007203674323685372-0.018806956440969965j,-0.04442425947959454-0.07249787051256527j],
                [0.0016257852270872-0.003053017794729039j,-0.0744360777594129-0.12780180683039016j],
                [-0.0901647284145893+0.0677340601480832j,0.0979636277930557-0.056366542766686j],  
                [-0.009266711197918743+0.027934965073992943j,0.02048992366395706+0.019193953602985394j],    
                [0.055799741897075504+0.004466873034050496j,-0.0019784118716146526-0.01152241830326015j]]

    coeffs2 = [ [0.07879274722101903+0.09754325107356299j],
                [0.31751612925781053+0.3421002694187953j],
                [0.14558393643440118+0.11628177802300182j],
                [-0.014307923079108272+0.01757723263834557j,0.06664885589712623-0.01162460678640483j],
                [-0.05585421324834267+0.10784840723070252j,0.07691593386321101-0.029235374950504007j],
                [0.05772597473348114-0.10096076943851984j,-0.024940720504176422+0.0758677528010174j],
                [0.15311623204411837-0.23609129713406637j,0.03861315831443918+0.05804037824648778j],
                [-0.10366089601851333-0.02172892293430301j,-0.11016024270244963+0.040068066693241415j],
                [-0.04414916142313346+0.037492297654245554j,-0.0031317556596735254+0.020400965203221797j],
                [-0.04468442844227466+0.08995652349993347j,0.07115329719988309+0.022450124394864823j]]

    flatten = lambda l: [item for sublist in l for item in sublist]

    fcoeffs = np.array(flatten(coeffs2))
    #print(np.sum(np.power(np.abs(fcoeffs),2)))

    v_order = [0,3,1,5,7,4,8,2,6,9]
    coeffs2 = [coeffs2[i] for i in v_order]
    #modes = [modes[i] for i in v_order]
    #print(modes)

    #modes = modes[:8]
    #coeffs = coeffs[:8]
    '''
    for mode,coeff in zip(modes,coeffs2):
        print(mode,np.power(np.abs(coeff),2))
        if mode[0]==0:
            field = normalize(LPmodes.lpfield(xg1,yg1,mode[0],mode[1],10,wl0,ncore,nclad))
            field1 += coeff[0]*field
            #plt.imshow(np.abs(field),extent=(-w1,w1,-w1,w1))
            #plt.show()
        else:
            field = normalize(LPmodes.lpfield(xg1,yg1,mode[0],mode[1],10,wl0,ncore,nclad))
            field1 += coeff[0]*field
            #plt.imshow(np.abs(field),extent=(-w1,w1,-w1,w1))
            #plt.show()

            field = normalize(LPmodes.lpfield(xg1,yg1,mode[0],mode[1],10,wl0,ncore,nclad,which='sin'))
            field1 += coeff[1]*field
            #plt.imshow(np.abs(field),extent=(-w1,w1,-w1,w1))
            #plt.show()
    

    print(np.power(overlap(u0,field1),2)/overlap(field1,field1))

    plt.imshow(np.abs(field1),extent=(-w1,w1,-w1,w1))
    plt.show()

    '''
    
    rcores = np.linspace(2,10,200)
    powers=[]
    ws=[]


    for rcore in rcores:
        V = LPmodes.get_V(k0,rcore,ncore,nclad)
        modes = LPmodes.get_modes(V)

        #res = minimize(get_power_in_modes_neg,100,args=(rcore,ncore,nclad,u0,modes,norm))
        w = res0[2]

        _power = get_power_in_modes(w,rcore,ncore,nclad,u0,modes,norm)
        powers.append(_power)
        ws.append(w)
        print(rcore,_power,w)

    data = np.array([rcores,powers,ws])

    np.save("gaussdata",data)

    np.save("psfabb_sr0pt1_lp01opt_6um",data)

    rcores, powers, ws = np.load("psfabb_sr0pt1_lp01opt_6um.npy")
    plt.plot(rcores,powers)
    #plt.plot(data[0],data[1])
    
    plt.xlabel("core rad (um)")
    plt.ylabel("coupling efficiency")

    plt.show()






def opt_at_wl(wl,tele,DMshapein,t_arr,ncore=1.4504,nclad=1.4449):

    #load DM shapes
    DMshapes = np.load(DMshapein+".npy")

    ### begin optimization segment ###

    wf_pupil = hc.Wavefront(tele.ap,wavelength = wl*1.e-6)

    wf_focus_perfect = tele.propagator.forward(wf_pupil)
    u_focus_perfect = get_u(wf_focus_perfect)

    plt.imshow(np.abs(u_focus_perfect))
    plt.show()

    rcore = 6.21
    k = 2*np.pi/wl
    V = LPmodes.get_V(k,rcore,ncore,nclad)
    modes = LPmodes.get_modes(V)

    #do the psf scale optimization at 6 um. pick a psf at "random" to do so.
    pick = 4800
    tele.atmos.t = t_arr[pick]
    tele.DM.actuators = DMshapes[pick]
    
    u = tele.get_PSF(wf_pupil)
    print(get_strehl(u,u_focus_perfect))
    plt.imshow(np.abs(u))
    plt.show()
    r,p,w = coupling.scale_u0_all(u,modes,rcore,ncore,nclad,wl)
    print('optimal psf size: ',w)

    return w

def compute_coupling_at_wl(wl,tele,w,DMshapein,t_arr,ncore=1.4504,nclad=1.4449):
    '''wl in um'''

    #load DM shapes
    DMshapes = np.load(DMshapein+".npy")

    rcore = 6.21
    k = 2*np.pi/wl
    V = LPmodes.get_V(k,rcore,ncore,nclad)
    modes = LPmodes.get_modes(V)
    
    ### begin coupling segment ###

    wf_pupil = hc.Wavefront(tele.ap,wavelength = wl*1.e-6)
    wf_focus_perfect = tele.propagator.forward(wf_pupil)
    u_focus_perfect = get_u(wf_focus_perfect)

    #now compute the couplings. atm only do it for 100 (or 99?) out of the 10000 PSFs.

    couplings = []

    for i in range(len(t_arr)):
        if i%100 == 0 and i!=0:
            tele.atmos.t = t_arr[i]
            tele.DM.actuators = DMshapes[i]
            
            #get the PSF
            u = tele.get_PSF(wf_pupil)

            strehl = get_strehl(u,u_focus_perfect)
            print("strehl", strehl)

            c = coupling.get_power_in_modes(w,rcore,ncore,nclad,u,modes,wl)
            couplings.append(c)

    ### end coupling segment ###

    return couplings

def get_num_modes(modes):
    count = 0
    for mode in modes:
        if mode[0]==0:
            count+=1
        else:
            count+=2
    return count

def get_IOR(wl):
    """ for fused silica """
    wl2 = wl*wl
    return np.sqrt(0.6961663 * wl2 / (wl2 - 0.0684043**2) + 0.4079426 * wl2 / (wl2 - 0.1162414**2) + 0.8974794 * wl2 / (wl2 - 9.896161**2) + 1)

def compute_coupling_data(psffile,ncore,nclad,rcore_ref,wl0,tag):

    header = "./couplingdata/"

    k0 = 2*np.pi/wl0

    u0 = np.load(psffile+".npy")
    u0 = normalize(resize(u0,(401,401)))

    V = LPmodes.get_V(k0,rcore_ref,ncore,nclad)
    modes = LPmodes.get_modes(V)

    opt_lp01 = scale_u0_lp01(u0,rcore_ref,ncore,nclad,wl0)
    opt_all = scale_u0_all(u0,modes,rcore_ref,ncore,nclad,wl0)

    rcores = np.linspace(2,10,200)
    powers_lp01 = []
    ws_lp01 = []
    powers_all = []
    ws_all = []

    w_lp01 = opt_lp01[2]
    w_all = opt_all[2]

    for rcore in rcores:
        V = LPmodes.get_V(k0,rcore,ncore,nclad)
        modes = LPmodes.get_modes(V)

        _p_lp01 = get_power_in_modes(w_lp01,rcore,ncore,nclad,u0,modes,wl0)
        _p_all  = get_power_in_modes(w_all,rcore,ncore,nclad,u0,modes,wl0)

        powers_lp01.append(_p_lp01)
        ws_lp01.append(w_lp01)
        powers_all.append(_p_all)
        ws_all.append(w_all)

    data_lp01 = np.array([rcores,powers_lp01,ws_lp01])
    data_all = np.array([rcores,powers_all,ws_all])

    fname_lp01 = header + psffile + "_lp01opt_" + tag
    fname_all = header + psffile + "_allopt_" + tag

    np.save(fname_lp01,data_lp01)
    np.save(fname_all,data_all)