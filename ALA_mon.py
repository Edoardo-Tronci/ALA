import torch
import numpy as np
import time

n_iter = 0

torch.set_default_dtype(torch.double)
is_cuda = torch.cuda.is_available()
#is_cuda = False
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def updatew(m,mj,W,f):
	#W,wmax = updatew(m,mj,W,f)
	W = np.roll(W,-1)
	W[m] = f.item()
	mj = min(mj+1,m)
	return W, max(W)

def compare_d(gs,gd,shs,dhd,iprint):
	if iprint >= 1:
		print('gs + 0.5*shs=',gs+0.5*shs)
		print('gd + 0.5*dhd=',gd+0.5*dhd)
	if (((gs+np.double(0.5)*shs) <= (gd+np.double(0.5)*dhd)) and (shs < 0.0)):
		choose_d = False
	else:
		choose_d = True

	return choose_d

def hessdirn(grad,n,x,s,g):
    snr = s.norm().item()
    #step = np.maximum(np.double(1.e-14)/snr,np.double(1.e-14))
    step = np.maximum(np.double(1.e-6)/np.double(snr),np.double(1.e-6))
    #print('hesdirn:   step=',step)
    with torch.no_grad():
            #print('hessdirn: ',x)
            #print('hessdirn: ',s)
            xp = (x + step*s)
            #xm = (x - step*s)
            #x1.requires_grad_()
    g1 = grad(xp)
    #g2 = grad(xm)
    #print('hesdirn:   x1=',x1)
    #print('hesdirn:   g1=',g1)
    #print('hesdirn: step=',step)
    hd = ((g1-g).float()/np.float(step)).double()
    return hd

def linesearch_newton(funct,n,x,d,gd,dhd,f,falfa1,\
                      f_DEC,f_iniz,delta_f0,k,nf,WMAX,xmn,g,\
                      ialfa1,inoposdef,isingular,itronc,igrad,iprint):

    x_norm = x.norm().item()
    d_norm = d.norm().item()
    g_norm = g.norm().item()
    gamma = np.double(1.e-3)
    sigma = np.double(0.5)
    epsm  = np.double(1.e-16)
    fail  = epsm*x_norm/max(1.0,d_norm)

#---------------------------------------------------
#   FW=f ---->  monotono  FW=WMAX ----> non monotono
#---------------------------------------------------
    if igrad or itronc:
        fw = f
    else:
        fw = WMAX

    alfa = np.double(1.0)

#106 continue
    while True:
        fr = fw + alfa * gamma * gd
        with torch.no_grad():
             z  = x  + alfa * d

        if (alfa == 1.0) and ialfa1:
            fz  = falfa1
        else:
            fz  = funct(z)
            nf += 1

        if alfa == 1.0:
            ftest = (fz - f_iniz)
            if (fz > f_iniz) and (ftest >= f_DEC*delta_f0):
                fw = f
                fr = fw + alfa * gamma * gd


        if iprint >= 1:
            print('FZ=',fz.item())
            print('FR=',fr.item())
            print('FW=',fw.item())
            print('WMAX=',WMAX)

        if fz <= fr:
            f  = fz
            with torch.no_grad():
                 x  = x + alfa * d
            k += 1
            fail_flag = False
            return x,f,alfa,nf,k,fail_flag
        else:
            if (fz-f)/min(1.e+12,max(1.e-12,-alfa*gamma*gd)) >= 1.e+10:
                aa = 2.e-12 * alfa
                a1 = 1.0 * max(1.0,x_norm)/max(1.e-15,d_norm)
                a1 = min(a1,sigma**2*alfa)
                alfa = max(aa,a1)
            else:
                alfa = sigma * alfa

    #115 continue
            if alfa <= fail:
                fail_flag = True
                print(' Linesearch failure GD=',gd,' GS=',gs)
                return x,f,alfa,nf,k,fail_flag

def linesearch_negcurv(funct,n,k,x,f,g,s,nf,gs,shs,alfa_old,\
                       delta_f_old,WMAX,neg_curv_first,s_norm_old,iprint, L_h):

    g_norm = g.norm().item()
    s_norm = s.norm().item()
    x_norm = x.norm().item()
    epsm  = np.double(1.e-16)
    fail  = epsm*x_norm/max(1.0,s_norm)
    gamma = np.double(1.e-3)
    sigma = np.double(0.5)
    fail_flag = False
    curtis = False
    f_iniz = f
    fw = f

    if neg_curv_first:
        neg_curv_first = False
        delta_f_old = np.abs(f.item())
        alfa_old = 1.0
        alfa_init = alfa_old
        s_norm_old = 1.0
    else:
        alfa_init = alfa_old * s_norm_old/s_norm
        s_norm_old = s_norm

    if iprint >= 1:
        print('gs=',gs)
        print('delta_f_old=',delta_f_old)

    alfa_init = (- gs - np.sqrt(np.power(gs,2) + np.double(2.0)*np.abs(delta_f_old)*np.abs(shs))) / shs
    beta_init = (- shs + np.sqrt(shs**2 - 2.0*L_h*s_norm**3 * gs)) / (L_h*s_norm**3)
#              write(*,*)'alfa_init =', alfa_init

#            alfa_init = DMIN1( alfa_init, 1.D+6 )
    alfa_init = min(alfa_init, np.double(1.e+3))
    alfa_init = max(alfa_init, np.double(1.e-6))
    # alfa = alfa_init

    if curtis:
        alfa = beta_init
    else:
        alfa = alfa_init
    # alfa = np.double(1.0)
    # alfa = beta_init
    if iprint >= 1:
        print('alfa_init curv. neg.=', alfa, ' f =',f.item())

    with torch.no_grad():
        z = x + alfa * s

    if curtis:
        fr = fw + gamma * alfa * (gs + 0.5 * alfa * shs + L_h*alfa**2.0*s_norm**3 / np.double(6.0))
    else:
        fr = fw + gamma * alfa * (gs + 0.5 * alfa * shs)


    fz = funct(z)
    nf += 1

    if iprint >= 1:
        print('fz_t. =',fz.item(),'        fr_t =',fr.item(),'        alfa =', alfa)

    if fz > fr:
        L_h = np.double(2.0)*L_h
# reduce the step
        while True:
            alfa = sigma * alfa

            if alfa <= fail:
                print(' alfa = ',alfa)
                print(' fail = ',fail)
                print(' (linesearch failure) ')
                flag_fail = True
                return x,f,alfa,nf,k,fail_flag,alfa_old,alfa,delta_f_old, L_h

            with torch.no_grad():
                z = x + alfa * s

            if curtis:
                fr = fw + gamma * alfa * (gs + 0.5 * alfa * shs + L_h*alfa**2.0*s_norm**3 / np.double(6.0))
            else:
                fr = fw + gamma * alfa * (gs + 0.5 * alfa * shs)


            fz = funct(z)
            nf += 1

            if iprint >= 1:
                print('fz_con =',fz.item(),'     fr_con =',fr.item(),'        alfa =', alfa)

            if fz <= fr:
                break

        f = fz
        with torch.no_grad():
            x = x + alfa * s
        k += 1
        alfa_old = alfa
        delta_f_old = f-f_iniz
        if iprint >= 1:
            print('final step: alpha=',alfa)
            print('f - f_iniz=',delta_f_old.item())

        flag_fail = False
        return x,f,alfa,nf,k,fail_flag,alfa_old,alfa,delta_f_old, L_h
    else:
        flag_req = True
        while True:
            f_app = fz
            with torch.no_grad():
                z = x + (alfa/sigma) * s

            if curtis:
                fr = fw + gamma * (alfa/sigma) * (gs + 0.5 * (alfa/sigma) * shs + L_h*(alfa/sigma)**2.0*s_norm**3 / 6.0)
            else:
                fr = fw + gamma * (alfa/sigma) * (gs + 0.5 * (alfa/sigma) * shs)

            fz = funct(z)
            nf += 1
            if iprint >= 1:
                print('fz_es. =',fz.item(),'        fr_es =',fr.item(),'        alfa =', alfa/sigma)

            if fz <= fr:
                if flag_req:
                    flag_req = False
                alfa = (alfa/sigma)
            else:
                break
        if not flag_req:
            L_h = L_h / np.double(2.0)
        with torch.no_grad():
            x = x + alfa * s
        f = f_app
        k += 1
        alfa_old = alfa
        delta_f_old = f - f_iniz
        if iprint >= 1:
            print('final step: alpha=',alfa)
            print('f - f_iniz=',delta_f_old)
        flag_fail = False

        return x,f,alfa,nf,k,fail_flag,alfa_old,alfa,delta_f_old, L_h

def dir(funct,grad,hessdir,n,k,x,g,ng,ninoposdef,nitronc,iprint,hd_exact,c,t):
    isingular = False
    inoposdef = False
    itronc    = False
    igrad     = False
    goth      = False
    eps       = np.double(1.e-6)

    #funct(x)

    dn        = torch.zeros(n,dtype=torch.double).to(device)
    dnbest    = torch.zeros(n,dtype=torch.double).to(device)
    dp        = torch.zeros(n,dtype=torch.double).to(device)
    d         = torch.zeros(n,dtype=torch.double).to(device)

    ndim      = dn.numel()
    #print(ndim)

    p         = -g.clone().detach()
    gq        =  g.clone().detach()
    gnr       = g.norm().item()
    gqnorm    = gnr

    fquad     = np.double(0.0)
    gd_new    = np.double(0.0)
    alfa      = np.double(0.0)
    alfabest  = np.double(0.0)
    max_rap   = np.double(0.0)
    min_rap   = np.inf

    deltatr   = np.double(1.e-3)*np.double(ndim)
    costz     = np.double(k)
    if k == 0:
        costz = np.double(1.0)
    costz1    = deltatr/costz
    delta     = min(gnr,costz1)
    kint      = 1
    goth      = False

    while True:
        if iprint >= 1:
            print('c.g. iteration ', kint)
        if kint == 1:
            pnr = gnr
            pnr2= gnr*gnr
        else:
            pnr = p.norm().item()
            pnr2= pnr*pnr

        p_norm = p/pnr

        if hd_exact:
            hd = hessdir(x,p_norm,goth)
            goth = True
        else:
            hd = hessdirn(grad,n,x,p_norm,g)

        ng += 1

        if (iprint >= 1):
            #print('dir: p_norm=',p_norm)
            print( 'dir: p_norm=',end='')
            for i in range(p_norm.numel()):
                print(p_norm[i].item(),' ',end='')
            print()
            print( 'dir:      g=',end='')
            for i in range(g.numel()):
                print(g[i].item(),' ',end='')
            print()
            print( 'dir:     hd=',end='')
            for i in range(hd.numel()):
                print(hd[i].item(),' ',end='')
            print()
            #input()

        #den = torch.dot(p_norm,hd).item()
        den = (p_norm*hd).sum().item()
        #print('p_norm=',p_norm)
        #print('hd=',hd)
        #print('den=',den)
        #print('pn2=',pnr2)
        #input()

        if eps*pnr2 <= np.double(1.e-16): #np.finfo(dtype=np.float).eps**2:
            if iprint >= 1:
                print(' * norm of s_k is too small *')
                print(' kint = ',kint)
                print('  pnorm^2 = ',pnr2,'  den = ',den)
            if kint == 1:
                igrad = True
                dp = -g.clone().detach()
            break
            #goto 6
        if np.abs(den) < eps:
            isingular = True
            if iprint >= 1:
                print(' * hessian matrix is singular *')
                print(' kint = ',kint)
            if kint == 1:
                igrad = True
                dp = -g.clone().detach()
            break
        rap = den
        max_rap = max(max_rap,rap)
        min_rap = min(min_rap,rap)

        #-------------------------
        #calcolo di alfa
        #-------------------------
        #print('dir: den=',den)
        #print('dir: pnr=',pnr)
        #print('dir:  gq=',gq)

        alfa_old = alfa
        #alfa = torch.dot(-(gq/pnr),p_norm).item()/den
        alfa = (-(gq/pnr)*p_norm).sum().item()/den
        alfa_norm = alfa*pnr

        if iprint >= 1:
            print('dir: alfa=',alfa)
            print('dir:  den=',den)
        #-----------------------------
        #calcolo nuova direzione
        #-----------------------------
        
        if den <= -eps:
            if c == "Curv":
                ninoposdef += 1
            else:
                break               # no curv
            with torch.no_grad():
                    dn += - alfa*p
            inoposdef   = True
            if np.abs(alfa) > alfabest:
                alfabest = alfa
                dnbest = p/gqnorm
        if den >= eps:
            #print('dir: dpold=',dp)
            #print('dir:  p=',p)
            #print('dir: alfa=',alfa)

            with torch.no_grad():
                    dp += alfa*p
            #print('dir: dpnew=',dp)

        gqnorm_old = gqnorm
        with torch.no_grad():
                d  += alfa*p
                gq += alfa*hd*pnr
        #print('d=',d)
        #print('alfa=',alfa)
        #print('pnr=',pnr)
        #print('hd=',hd)
        #print('gq=',gq)
        fquad_old = fquad
        gd_old = gd_new
        #gd_new = torch.dot(g,d).item()
        #fquad = torch.dot(gq,d).item()
        #fquad = (fquad + torch.dot(g,d).item())/np.double(2.0)

        gd_new = (g*d).sum().item()
        fquad = (gq*d).sum().item()
        fquad = (fquad + (g*d).sum().item())/np.double(2.0)
        #-----------------------------
        #criterio di troncamento
        #-----------------------------
        gqnorm = gq.norm().item()
        dnr    = d.norm().item()
        test   = gnr*delta
        if dnr < gnr:
            tolnew = np.double(10.e-4)*min(np.double(1.0),dnr)*np.sqrt(np.double(ndim))
        else:
            tolnew = np.double(10.e-4)*np.sqrt(np.double(ndim))

        criterio = np.abs((-np.double(3.0/2.0)*(gd_new-gd_old)+fquad-fquad_old)/(-np.double(3.0/2.0)*gd_new+fquad))*np.double(kint)
        if iprint >= 1:
            print('  tolnew= %25.16e' % tolnew)
            print('criterio= %25.16e' % criterio)
        if ((t == "Nash") and (criterio > tolnew) and (kint <= 20)) or ((t == "Dembo") and (gqnorm > test)):
            #-----------------------------
            #criterio non soddisfatto
            #-----------------------------
            if kint == 2*ndim:
                itronc   = True
                nitronc += 1
                if iprint >= 2:
                    print(' * conjugate gradient failed *')
                    print(' kint = ',kint)
                    print('  GRAD = ',gqnorm,'  test = ',test)
                    break

            #beta      = torch.dot(gq/pnr,hd).item()/den
            beta      = ((gq/pnr)*hd).sum().item()/den
            beta_norm = beta*pnr
            with torch.no_grad():
                    p         = -gq + beta*p
                    #print('p=',p)
            #gp        = torch.dot(g,p).item()
            #gqp       = torch.dot(gq,p).item()
            gp        = (g*p).sum().item()
            gqp       = (gq*p).sum().item()

            if np.abs(gqp-gp) > np.double(1.e-3)*(np.abs(gqp)+np.double(1.e-6)):
                    if iprint >= 1:
                        print('fallimento coniugatezza iterazione 1 =',kint)
                        print('gp= %25.16e gqp= %25.16e gqnorm= %25.16e' % (gp,gqp,gqnorm))
                    break

            if ((gp*gqp < 0.0) and (np.abs(gqp-gp) > np.double(1.e-6)*(np.abs(gqp)+np.double(1.e-9)))):
                    if iprint >= 1:
                        print('fallimento coniugatezza iterazione 2 =',kint)
                        print('gp= %25.16e gqp= %25.16e gqnorm= %25.16e' % (gp,gqp,gqnorm))
                    break

            kint += 1
            if (iprint >= 1):
                print('dir:  ng=',ng)
                #input()
        else:
                break
        #end while

    #6 continue

    d = dp.clone().detach()
    dnr = d.norm().item()
    #gd = torch.dot(g,d)
    gd = (g*d).sum()
    if dnr > np.double(0.0):
        if hd_exact:
            hd = hessdir(x,d,goth)
        else:
            hd = hessdirn(grad,n,x,d,g)
        #print('x=',x)
        #print('d=',d)
        #print('g=',g)
        #print('hd=',hd)
        #dhd = torch.dot(d,hd).item()
        #gd = torch.dot(g,d).item()
        dhd = (d*hd).sum().item()
        gd = (g*d).sum().item()
        if iprint >= 1:
            print(' pos.curv. direction  gs = %25.16e' % gd)
            print(' pos.curv. direction shs = %25.16e' % dhd)
    else:
        dhd = 0.0

    if inoposdef:
        with torch.no_grad():
            s = dn + dnbest
        if ninoposdef > 1:
            if iprint >= 1:
                print(' inoposdef = ',inoposdef)
            snr = s.norm().item()
            if hd_exact:
                hd = hessdir(x,s,goth)
            else:
                hd = hessdirn(grad,n,x,s,g)
            ng += 1
            #shd = torch.dot(s,hd).item()
            shd = (s*hd).sum().item()
            if shd >= 0.0:
                if iprint >= 1:
                    print(' neg.curv. direction',shd)
                    print('dn+dnbest is not a neg.curv. direction')
                s = dn.clone().detach()

        snr = s.norm().item()
        #gs  = torch.dot(g,s).item()
        gs  = (g*s).sum().item()
        if gs > 0.0:
            s = -s
            gs = -gs

        if hd_exact:
            hd = hessdir(x,s,goth)
        else:
            hd = hessdirn(grad,n,x,s,g)
        #shs = torch.dot(s,hd).item()
        shs = (s*hd).sum().item()
        if iprint >= 1:
            print(' neg.curv. direction  gs = %25.16e' % gs)
            print(' neg.curv. direction shs = %25.16e' % shs)
    else:
        s = torch.zeros(n,dtype=torch.double).to(device)
        snr = np.double(0.0)
        gs  = np.double(0.0)
        shs = np.double(0.0)

    return d,s,gd,gs,shs,dhd,dnr,snr,kint,ng,inoposdef,isingular,itronc,igrad,ninoposdef



def NWTNM(funct,grad,hessdir,x,gmin,maxls,max_time,iprint,satura,hd_exact,name,r,nneu,c,t):
	n          = x.shape
	ndim       = x.numel()
	m          = 0 #0 #20 #100
	nmax       = 0 #0 #20
	nitertot   = 0
	ninoposdef = 0
	nitronc    = 0
	kint       = 0
	ktot       = 0
	nf         = 0
	ng         = 0
	s_norm_old = np.double(0.0)
	alfa_old   = np.double(0.0)
	delta_f_old= np.double(0.0)
	xmn        = x.clone().detach()
	f_dec      = np.double(1.e+6)
	beta       = np.double(0.9)
	f          = funct(x)
	g0         = grad(x)
	g0max      = g0.norm(p=float("inf")).item()
	fold       = f + 10.0
	ifunct     = True
	nf        += 1
	f_min      = f
	f_max      = f
	delta0     = np.double(0.0) #np.double(0.0) #np.double(1.e+3)
	W          = [f.item() for i in range(m+1)]
	delta_f0   = np.double(0.0)
	L_h        = np.double(1.0)
	if iprint >= 0:
		print(' ')

	start_time = time.time()

	while True:   ## 100 continue
		if (ktot > 0):
			if iprint >= 1:
				print("------- RESTARTING -------")

		k = 0
		lj = 0
		mj = 0
		nn = 0
		W[m] = f.item()
		wmax = f.item()
		newton_first = True
		neg_curv_first = True

		while True:   ## 200 continue
			ialfa1 = 0
			falfa1 = np.nan
			if ifunct:
				if f < f_min:
					f_min = f
					xmn = x.clone().detach()
			g = grad(x)
			#print('NWTNM: x=',x)
			#print('NWTNM: g=',g)
			ng += 1
			g_norm = g.norm().item()

# 			if iprint >= 0:
# 				print(' K=%5d  KTOT=%5d  Nf=%5d  Ng=%6d  Nint=%9d  f=%12.5e  g_norm=%12.5e time=%9.2f' % (k,ktot+k,nf,ng,nitertot,f,g_norm,time.time()-start_time))
			global n_iter
			n_iter = ktot+k

# 			if iprint >= 1:
# 				print('')
# 				print(' K=%5d  KTOT=%5d  Nf=%5d  Ng=%6d  Nint=%9d  f=%12.5e  g_norm=%12.5e' % (k,ktot+k,nf,ng,nitertot,f,g_norm))
# 				print('')
                
			if iprint >= 2:
				print('x=',end='')
				for i in range(x.numel()):
					print(x[i].item(),' ',end='')
				print()

			if iprint >= 1:
				print('ng=',ng)

			d,s,gd,gs,shs,dhd,d_norm,s_norm,kint,ng,inoposdef,isingular,itronc,igrad,ninoposdef = dir(funct,grad,hessdir,n,k,x,g,ng,ninoposdef,nitronc,iprint,hd_exact,c,t)

			if iprint >= 1:
				print('d,s,g=',end='')
				for i in range(d.numel()):
					print(d[i].item(),' ',s[i].item(),' ',g[i].item())
				print()
			if iprint >= 1:
				print('ng=',ng)
				#input()

			nitertot += kint
			gmax = g.norm(p=float("inf")).item()

			##############################################
			# CRITERIO DI ARRESTO
			##############################################

			if ((gmax <= gmin) or ((d_norm+s_norm) <= 1.e-16) or (k+ktot > maxls) or (time.time()-start_time > max_time) or
				( not satura and ((torch.abs(f-fold)/torch.abs(fold) <= 1.e-2) or (gmax <= 1.e-1*g0max)) ) ):
				if (iprint >= 0):
					fid = open(t + "_" + c + '.txt','a')		
					print('%13s & %5d & %5d & %5d & %5d & %5d & %6d & %9d & %12.5e & %12.5e & %9.2f\\' % (name,r,nneu,k,ktot+k,nf,ng,nitertot,f,g_norm,time.time()-start_time),file=fid)
					fid.close()
# 					print('\n    ==================================================')
# 					print(  '    CRITERIO DI ARRESTO SODDISFATTO CON NORMA INFINITO')
# 					print(  '    ==================================================')
# 					print('     f = %13.6e   gmin = %13.6e   gnr = %13.6e   gmax = %13.6e' % (f,gmin,g_norm,gmax))
# 					print('     f = %13.6e   fold = %13.6e   f-fold/fold = %13.6e' % (f,fold,torch.abs(f-fold)/torch.abs(fold)))
# 					print("negative curvature:", ninoposdef)
                    
				if not ifunct:
					f = funct(x)
					nf += 1
					ifunct = True

				if f.item() <= f_min.item() + 1.e-6*np.abs(f_min.item()):
					return  f,x,k+ktot,nf,ng,ninoposdef,time.time()-start_time 
				else:
					#restart from the best point
					if iprint > 0:
						print(' restart from best point')
					x = xmn.clone().detach()
					f = f_min
					ktot += k
					ng -= 1
					delta0 = delta0/10.0
					W = [-np.inf for i in range(m+1)]
					wmax = W[0]
					m = int((m+1)/5)
					break  # goto 100

			fold = f

			if k == 0:
				f_iniz = f

			choose_d = compare_d(gs,gd,shs,dhd,iprint)

			if not choose_d:
				if not (k == lj):
					if not ifunct:
						f = funct(x)
						nf += 1
						ifunct = True
					if f > wmax:
						x = xmn.clone().detach()
						f = f_min
						ktot += k
						ng -= 1
						delta0 = delta0/10.0
						W = [-np.inf for i in range(m+1)]
						wmax = W[0]
						m = int((m+1)/5)
						break # goto 100
					else:
						if f < f_min:
							f_min = f
							xmn = x.clone().detach()
						W,wmax = updatew(m,mj,W,f)
						lj = k

				x,f,alfa,nf,k,fail_flag,alfa_old,alfa,delta_f_old, L_h = linesearch_negcurv(funct,n,k,x,f,g,s,nf,gs,shs,alfa_old,delta_f_old,wmax,neg_curv_first,s_norm_old,iprint, L_h)

				if f < f_min:
					f_min = f
					xmn = x.clone().detach()
				W,wmax = updatew(m,mj,W,f)
				lj = k
				#GOTO 200
			else:
				#print('   flags= ',newton_first, isingular, itronc, igrad)
				if (not newton_first) and (not ((isingular) or (itronc) or (igrad))):
					#print('k=',k,' lj=',lj,' nn=',nn)
					if (k == lj+nn):
						if not ifunct:
							f = funct(x)
							nf += 1
							ifunct = True
						nn = min(nn+ndim,nmax)
						if iprint >= 1:
							print('f=',f.item(),' wmax=',wmax)
						if f > wmax:
							if iprint >= 1:
								print(' control point')
							x = xmn.clone().detach()
							f = f_min
							ktot += k
							ng -= 1
							delta0 = delta0/10.0
							W = [-np.inf for i in range(m+1)]
							wmax = W[0]
							m = int((m+1)/5)
							break
						else:
							if f < f_min:
								f_min = f
								xmn = x.clone().detach()
							W,wmax = updatew(m,mj,W,f)
							lj = k

					if iprint >= 1:
						print('d_norm=',d_norm,' DELTA=',delta)
						print('f=',f.item(),' wmax=',wmax)
					if iprint >= 2:
						print('x=',end='')
						for i in range(x.numel()):
							print(x[i].item(),' ',end='')
						print()
						print('d=',end='')
						for i in range(d.numel()):
							print(d[i].item(),' ',end='')
						print()
					if d_norm <= delta:
						delta = delta*beta
						with torch.no_grad():
							v = x + d
						if iprint >= 1:
							print('d_norm_old=',d_norm_old)
						if d_norm <= d_norm_old:
							k += 1
							x = v.clone().detach()
							if iprint >= 1:
								print('accetto passo 1')
							if iprint >= 2:
								print('x=',end='')
								for i in range(x.numel()):
									print(x[i].item(),' ',end='')
								print()
							ifunct = False
							f = funct(x)
							d_norm_old = d_norm
							#########  GOTO 200
							continue

						fv = funct(v)
						nf += 1
						ialfa1 = 1
						falfa1 = fv
						ftest = fv - f_iniz_newton

						#print('fv=',fv,' f_iniz_newton=',f_iniz_newton,' ftest=',ftest,' f_dec=',f_dec,' delta_f0=',delta_f0)
						if (fv < f_iniz_newton) or (ftest <= f_dec*delta_f0):
							f = fv
							k += 1
							x = v.clone().detach()
							ifunct = True
							d_norm_old = d_norm
							########## GOTO 200
							continue

				if iprint >= 1:
					print(' steplength rejected')

				if not (k == lj):
					if not ifunct:
						f = funct(x)
						nf += 1
						ifunct = True
					if f > wmax:
						x = xmn.clone().detach()
						f = f_min
						ktot += k
						ng -= 1
						delta0 = delta0/10.0
						W = [-np.inf for i in range(m+1)]
						wmax = W[0]
						m = int((m+1)/5)
						break
					else:
						if f < f_min:
							f_min = f
							xmn = x.clone().detach()
						W,wmax = updatew(m,mj,W,f)
						lj = k

				if newton_first:
					f_iniz_newton = f

				x,f,alfa,nf,k,fail_flag = linesearch_newton(funct,n,x,d,gd,dhd,f,falfa1,\
                      f_dec,f_iniz,delta_f0,k,nf,wmax,xmn,g,ialfa1,inoposdef,isingular,itronc,igrad,iprint)

				if f < f_min:
					f_min = f
					xmn = x.clone().detach()

				W,wmax = updatew(m,mj,W,f)
				lj = k

				if newton_first:
					delta_f0 = f_iniz_newton - f
					delta = delta0 * alfa * d_norm
					newton_first = False

				d_norm_old = d_norm

				#GOTO 200

			#end if not choose_d

                #end while 200

        #end while 100
