package spring.generated.testconfiguration.controller;

import java.lang.String;
import javax.servlet.http.HttpServletRequest;
import javax.validation.Valid;

import org.hibernate.HibernateException;
import org.hibernate.Transaction;
import org.springframework.stereotype.Controller;
import org.springframework.ui.ModelMap;
import org.springframework.validation.BindingResult;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.servlet.ModelAndView;

import spring.mine.common.controller.BaseController;
import spring.mine.common.form.BaseForm;
import spring.mine.common.validator.BaseErrors;
import spring.generated.forms.PanelRenameEntryForm;
import us.mn.state.health.lims.common.services.DisplayListService;

import us.mn.state.health.lims.hibernate.HibernateUtil;
import us.mn.state.health.lims.localization.daoimpl.LocalizationDAOImpl;
import us.mn.state.health.lims.localization.valueholder.Localization;
import us.mn.state.health.lims.panel.dao.PanelDAO;
import us.mn.state.health.lims.panel.daoimpl.PanelDAOImpl;
import us.mn.state.health.lims.panel.valueholder.Panel;

@Controller
public class PanelRenameEntryController extends BaseController {
	@RequestMapping(
			value = "/PanelRenameEntry", 
			method = RequestMethod.GET
	)
	public ModelAndView showPanelRenameEntry(HttpServletRequest request,
			@ModelAttribute("form") PanelRenameEntryForm form) {	
		String forward = FWD_SUCCESS;
		if (form == null ) {
			 form = new PanelRenameEntryForm();
		}
		form.setFormAction("");
		BaseErrors errors = new BaseErrors();
		ModelAndView mv = checkUserAndSetup(form, errors, request);

		if (errors.hasErrors()) {
			return mv;
		}

		form.setPanelList(DisplayListService.getList(DisplayListService.ListType.PANELS));

		return findForward(forward, form);
	}

	protected ModelAndView findLocalForward(String forward, BaseForm form) {
		if (FWD_SUCCESS.equals(forward)) {
			return new ModelAndView("panelRenameDefinition", "form", form);
		} else if (FWD_SUCCESS_INSERT.equals(forward)) {
		      return new ModelAndView("redirect:/PanelRenameEntry.do", "form", form);
		} else {
			return new ModelAndView("PageNotFound");
		}
	}
	
	@RequestMapping(
			value = "/PanelRenameEntry", 
			method = RequestMethod.POST
		)
	public ModelAndView updatePanelRenameEntry(HttpServletRequest request,
			@ModelAttribute("form") PanelRenameEntryForm form) {	
		
		String forward = FWD_SUCCESS_INSERT;

		String panelId = form.getPanelId();
		String nameEnglish = form.getNameEnglish();
		String nameFrench = form.getNameFrench();
		String userId = getSysUserId(request);

		updatePanelNames(panelId, nameEnglish, nameFrench, userId);

		form = new PanelRenameEntryForm();
		form.setFormAction("");

		form.setPanelList(DisplayListService.getList(DisplayListService.ListType.PANELS));

		return findForward(forward, form);
	}

	private void updatePanelNames(String panelId, String nameEnglish, String nameFrench, String userId) {
		PanelDAO panelDAO = new PanelDAOImpl();
		Panel panel = panelDAO.getPanelById(panelId);
		
		if (panel != null) {

			Localization name = panel.getLocalization();
			name.setEnglish(nameEnglish.trim());
			name.setFrench(nameFrench.trim());
			name.setSysUserId(userId);

			Transaction tx = HibernateUtil.getSession().beginTransaction();

			try {
				new LocalizationDAOImpl().updateData(name);
				tx.commit();
			} catch (HibernateException e) {
				tx.rollback();
			} finally {
				HibernateUtil.closeSession();
			}

		}

		// Refresh panel names
		DisplayListService.getFreshList(DisplayListService.ListType.PANELS);
	}

	protected String getPageTitleKey() {
		return null;
	}

	protected String getPageSubtitleKey() {
		return null;
	}
}
